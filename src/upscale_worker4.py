import sys
import os
import traceback
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Model, Input
from PIL import Image

import math
import tensorflow as tf
from tensorflow.keras import layers

class ChannelAttention(layers.Layer):
    """
    Squeeze-and-Excitation channel recalibration.
    reduction=4  →  bottleneck = 64 // 4 = 16 channels  (was 4 with reduction=16)
    """
    def __init__(self, channels, reduction=4, **kwargs):
        super().__init__(**kwargs)
        bottleneck  = max(channels // reduction, 4)
        self.gap    = layers.GlobalAveragePooling2D(keepdims=True)
        self.conv1  = layers.Conv2D(bottleneck, 1, activation='gelu')
        self.conv2  = layers.Conv2D(channels,   1, activation='sigmoid')

    def call(self, x):
        return x * self.conv2(self.conv1(self.gap(x)))


class SwinBlock(layers.Layer):
    """
    Complete Swin Transformer block.

    Set use_shift=False for W-MSA and use_shift=True for SW-MSA.
    Always alternate W/SW pairs in the generator (see build_ultimate_generator).
    """
    def __init__(self, window_size=8, num_heads=3, channels=64,
                 use_shift=False, mlp_ratio=4.0, **kwargs):
        super().__init__(**kwargs)
        self.window_size = window_size
        self.shift_size  = window_size // 2 if use_shift else 0
        self.use_shift   = use_shift
        self.num_heads   = num_heads
        self.channels    = channels
        self.scale       = (channels // num_heads) ** -0.5

        self.norm1  = layers.LayerNormalization(epsilon=1e-5, dtype='float32')
        self.qkv    = layers.Dense(channels * 3, use_bias=True)
        self.proj   = layers.Dense(channels)

        ws = window_size
        self.rel_pos_table = self.add_weight(
            shape=((2 * ws - 1) ** 2, num_heads),
            initializer='zeros', trainable=True,
            name='rel_pos_bias_table'
        )
        import numpy as _np
        ch = _np.arange(ws); cw = _np.arange(ws)
        coords     = _np.stack(_np.meshgrid(ch, cw, indexing='ij'))
        flat       = coords.reshape(2, -1)
        rel        = flat[:, :, None] - flat[:, None, :]
        rel        = rel.transpose(1, 2, 0)
        rel[:, :, 0] += ws - 1
        rel[:, :, 1] += ws - 1
        rel[:, :, 0] *= 2 * ws - 1
        self.rel_pos_index = tf.constant(
            (rel[:, :, 0] + rel[:, :, 1]).astype(_np.int32))

        mlp_hidden   = int(channels * mlp_ratio)
        self.norm2   = layers.LayerNormalization(epsilon=1e-5, dtype='float32')
        self.ffn_fc1 = layers.Dense(mlp_hidden, activation='gelu')
        self.ffn_fc2 = layers.Dense(channels)

    def _partition(self, x, ws):
        B = tf.shape(x)[0]; H = tf.shape(x)[1]; W = tf.shape(x)[2]
        C = x.shape[-1]
        x = tf.reshape(x, (B, H // ws, ws, W // ws, ws, C))
        x = tf.transpose(x, (0, 1, 3, 2, 4, 5))
        return tf.reshape(x, (-1, ws * ws, C))

    def _reverse(self, wins, ws, H, W, C):
        B = tf.shape(wins)[0] // ((H // ws) * (W // ws))
        x = tf.reshape(wins, (B, H // ws, W // ws, ws, ws, C))
        x = tf.transpose(x, (0, 1, 3, 2, 4, 5))
        return tf.reshape(x, (B, H, W, C))

    def _shift_mask(self, H, W):
        ws  = self.window_size
        ss  = self.shift_size
        Hf  = tf.cast(H, tf.float32)
        Wf  = tf.cast(W, tf.float32)
        h   = tf.cast(tf.range(H), tf.float32)
        w   = tf.cast(tf.range(W), tf.float32)

        h_lbl = (tf.cast(h >= (Hf - tf.cast(ws, tf.float32)), tf.float32)
               + tf.cast(h >= (Hf - tf.cast(ss, tf.float32)), tf.float32))
        w_lbl = (tf.cast(w >= (Wf - tf.cast(ws, tf.float32)), tf.float32)
               + tf.cast(w >= (Wf - tf.cast(ss, tf.float32)), tf.float32))

        label_map = (tf.reshape(h_lbl, (H, 1)) * 3.0
                   + tf.reshape(w_lbl, (1, W)))
        label_map = tf.reshape(label_map, (1, H, W, 1))

        win_lbl = self._partition(label_map, ws)
        win_lbl = tf.reshape(win_lbl, (-1, ws * ws))

        diff = (tf.expand_dims(win_lbl, 2)
              - tf.expand_dims(win_lbl, 1))
        mask = tf.cast(tf.not_equal(diff, 0.0), tf.float32) * -100.0
        return tf.expand_dims(mask, 1)

    def call(self, x, training=None):
        B   = tf.shape(x)[0]
        H   = tf.shape(x)[1]
        W   = tf.shape(x)[2]
        C   = self.channels
        ws  = self.window_size
        ws2 = ws * ws
        shortcut = x

        x_n = tf.cast(self.norm1(tf.cast(x, tf.float32)), x.dtype)

        if self.use_shift:
            ss  = self.shift_size
            x_n = tf.roll(x_n, shift=[-ss, -ss], axis=[1, 2])
            mask = self._shift_mask(H, W)
        else:
            mask = None

        wins    = self._partition(x_n, ws)
        nW_tot  = tf.shape(wins)[0]

        qkv = self.qkv(wins)
        qkv = tf.reshape(
            qkv, (nW_tot, ws2, 3, self.num_heads, C // self.num_heads))
        qkv = tf.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = tf.matmul(q, k, transpose_b=True) * self.scale

        pb   = tf.gather(self.rel_pos_table,
                         tf.reshape(self.rel_pos_index, [-1]))
        pb   = tf.reshape(pb, (ws2, ws2, self.num_heads))
        pb   = tf.transpose(pb, (2, 0, 1))
        attn = attn + tf.expand_dims(pb, 0)

        if mask is not None:
            nW_count = (H // ws) * (W // ws)
            a4   = tf.reshape(attn, (B, nW_count, self.num_heads, ws2, ws2))
            a4   = a4 + tf.reshape(mask, (1, nW_count, 1, ws2, ws2))
            attn = tf.reshape(a4, (B * nW_count, self.num_heads, ws2, ws2))

        attn  = tf.nn.softmax(attn, axis=-1)
        x_out = tf.matmul(attn, v)
        x_out = tf.transpose(x_out, (0, 2, 1, 3))
        x_out = tf.reshape(x_out, (nW_tot, ws2, C))
        x_out = self.proj(x_out)

        x_rev = self._reverse(x_out, ws, H, W, C)
        if self.use_shift:
            ss    = self.shift_size
            x_rev = tf.roll(x_rev, shift=[ss, ss], axis=[1, 2])

        x = shortcut + x_rev

        shortcut2 = x
        x_n2 = tf.cast(self.norm2(tf.cast(x, tf.float32)), x.dtype)
        x_n2 = self.ffn_fc2(self.ffn_fc1(x_n2))
        x    = shortcut2 + x_n2

        return x

class TFSS2DMambaBlock(layers.Layer):
    def __init__(
        self,
        dim,
        state_dim=16,
        expansion=2,
        dt_rank=None,
        res_scale=1.0,
        dropout=0.0,
        use_channel_attention=True,
        ca_reduction=4,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.dim = int(dim)
        self.state_dim = int(state_dim)
        self.expansion = int(expansion)
        self.inner_dim = int(expansion * dim)
        self.dt_rank = int(dt_rank) if dt_rank is not None else math.ceil(self.dim / 16)
        self.res_scale = float(res_scale)
        self.dropout_rate = float(dropout)
        self.use_channel_attention = use_channel_attention

        self.norm = layers.LayerNormalization(epsilon=1e-5)
        self.in_proj = layers.Dense(self.inner_dim * 2, use_bias=False)

        self.local_conv = layers.DepthwiseConv2D(
            kernel_size=3, padding="same",
            depth_multiplier=1, activation=tf.nn.silu, use_bias=True,
        )

        self.x_proj = layers.Dense(self.dt_rank + 2 * self.state_dim, use_bias=False)
        self.dt_proj = layers.Dense(self.inner_dim, use_bias=True)
        self.out_proj = layers.Dense(self.dim, use_bias=False)
        self.dropout = layers.Dropout(self.dropout_rate) if self.dropout_rate > 0 else None

        if self.use_channel_attention:
            mid = max(self.inner_dim // ca_reduction, 4)
            self.ca_fc1 = layers.Dense(mid, activation="silu", use_bias=True)
            self.ca_fc2 = layers.Dense(self.inner_dim, use_bias=True)

    def build(self, input_shape):
        import numpy as np
        A_init = np.log(
            np.tile(np.arange(1, self.state_dim + 1, dtype=np.float32),
                    (self.inner_dim, 1))
        )
        self.A_log = self.add_weight(
            name="A_log",
            shape=(self.inner_dim, self.state_dim),
            initializer=tf.keras.initializers.Constant(A_init),
            trainable=True,
        )
        self.D = self.add_weight(
            name="D", shape=(self.inner_dim,),
            initializer="ones", trainable=True,
        )
        super().build(input_shape)

    def _flatten_hw(self, x, order="row"):
        if order == "row":
            return tf.reshape(x, [tf.shape(x)[0], -1, tf.shape(x)[3]])
        else:
            x = tf.transpose(x, [0, 2, 1, 3])
            return tf.reshape(x, [tf.shape(x)[0], -1, tf.shape(x)[3]])

    def _unflatten_hw(self, x, h, w, order="row"):
        b, c = tf.shape(x)[0], tf.shape(x)[2]
        if order == "row":
            return tf.reshape(x, [b, h, w, c])
        else:
            return tf.transpose(tf.reshape(x, [b, w, h, c]), [0, 2, 1, 3])

    def _parallel_selective_scan_1d(self, u, delta, B, C):
        A = -tf.exp(self.A_log)

        log_dA = tf.einsum("bld,dn->bldn", delta, A)
        log_dA_cumsum = tf.math.cumsum(log_dA, axis=1)
        dA_cumsum = tf.exp(log_dA_cumsum)

        dB = tf.einsum("bld,bln->bldn", delta, B)
        input_contrib = dB * u[..., None]

        safe_dA = dA_cumsum + 1e-12
        scaled = input_contrib / safe_dA
        prefix = tf.math.cumsum(scaled, axis=1)
        h = dA_cumsum * prefix

        y = tf.einsum("bldn,bln->bld", h, C)
        y = y + u * self.D[None, None, :]
        return y

    def _scan_direction(self, u_2d, delta_2d, B_2d, C_2d, order="row", reverse=False):
        u = self._flatten_hw(u_2d, order)
        delta = self._flatten_hw(delta_2d, order)
        B = self._flatten_hw(B_2d, order)
        C = self._flatten_hw(C_2d, order)

        if reverse:
            u, delta, B, C = [tf.reverse(t, axis=[1]) for t in [u, delta, B, C]]

        y = self._parallel_selective_scan_1d(u, delta, B, C)

        if reverse:
            y = tf.reverse(y, axis=[1])

        return self._unflatten_hw(y, tf.shape(u_2d)[1], tf.shape(u_2d)[2], order)

    def call(self, x, training=None):
        shortcut = x
        x = self.norm(x)

        x_proj = self.in_proj(x)
        u, z = tf.split(x_proj, 2, axis=-1)
        u = self.local_conv(u)

        b, h, w = tf.shape(u)[0], tf.shape(u)[1], tf.shape(u)[2]
        u_flat = tf.reshape(u, [b, h * w, self.inner_dim])

        params = self.x_proj(u_flat)
        dt_raw, B, C = tf.split(
            params, [self.dt_rank, self.state_dim, self.state_dim], axis=-1
        )
        delta = tf.nn.softplus(self.dt_proj(dt_raw))

        u_2d = tf.reshape(u_flat, [b, h, w, self.inner_dim])
        delta_2d = tf.reshape(delta, [b, h, w, self.inner_dim])
        B_2d = tf.reshape(B, [b, h, w, self.state_dim])
        C_2d = tf.reshape(C, [b, h, w, self.state_dim])

        y1 = self._scan_direction(u_2d, delta_2d, B_2d, C_2d, "row", False)
        y2 = self._scan_direction(u_2d, delta_2d, B_2d, C_2d, "row", True)
        y3 = self._scan_direction(u_2d, delta_2d, B_2d, C_2d, "col", False)
        y4 = self._scan_direction(u_2d, delta_2d, B_2d, C_2d, "col", True)

        y = (y1 + y2 + y3 + y4) / 4.0

        if self.use_channel_attention:
            gap = tf.reduce_mean(y, axis=[1, 2], keepdims=True)
            ca = tf.nn.sigmoid(self.ca_fc2(self.ca_fc1(gap)))
            y = y * ca

        y = y * tf.nn.silu(z)
        y = self.out_proj(y)

        if self.dropout is not None:
            y = self.dropout(y, training=training)

        return shortcut + self.res_scale * y
    

class DenseBlock(layers.Layer):
    def __init__(self, channels=64, growth_channels=32, **kwargs):
        super().__init__(**kwargs)
        gc = growth_channels
        self.c1 = layers.Conv2D(gc, 3, padding='same', activation='gelu',
                                kernel_initializer='he_normal')
        self.c2 = layers.Conv2D(gc, 3, padding='same', activation='gelu',
                                kernel_initializer='he_normal')
        self.c3 = layers.Conv2D(gc, 3, padding='same', activation='gelu',
                                kernel_initializer='he_normal')
        self.c4 = layers.Conv2D(gc, 3, padding='same', activation='gelu',
                                kernel_initializer='he_normal')
        self.c5 = layers.Conv2D(channels, 3, padding='same',
                                kernel_initializer='he_normal')

    def call(self, x):
        x1 = self.c1(x)
        x2 = self.c2(tf.concat([x, x1],             axis=-1))
        x3 = self.c3(tf.concat([x, x1, x2],         axis=-1))
        x4 = self.c4(tf.concat([x, x1, x2, x3],     axis=-1))
        x5 = self.c5(tf.concat([x, x1, x2, x3, x4], axis=-1))
        return x5 * 0.2 + x


class RRDB(layers.Layer):
    def __init__(self, channels=64, growth_channels=32, **kwargs):
        super().__init__(**kwargs)
        self.rdb1 = DenseBlock(channels, growth_channels)
        self.rdb2 = DenseBlock(channels, growth_channels)
        self.rdb3 = DenseBlock(channels, growth_channels)
        self.ca   = ChannelAttention(channels, reduction=4)

    def call(self, x):
        out = self.rdb3(self.rdb2(self.rdb1(x)))
        out = self.ca(out)
        return out * 0.2 + x


def upsample_block(x, channels, scale=2):
    x = layers.Conv2D(channels * (scale ** 2), 3, padding='same',
                      kernel_initializer='he_normal')(x)
    x = layers.Lambda(lambda z: tf.nn.depth_to_space(z, block_size=scale))(x)
    x = layers.Activation('gelu')(x)
    x = layers.Conv2D(channels, 3, padding='same',
                      kernel_initializer='he_normal')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    return x

def build_ultimate_generator1(input_shape=(None, None, 3),
                              num_rrdb=12, channels=64, scale=2,
                              window_size=8, num_heads=2):
    inputs = Input(shape=input_shape)

    x = layers.Conv2D(channels, 3, padding='same',
                       kernel_initializer='he_normal', name='init_conv')(inputs)
    x = layers.LeakyReLU(alpha=0.2, name='init_lrelu')(x)
    feat_init = x

    feat = feat_init
    swin_idx = 0
    mamba_idx = 0

    for i in range(num_rrdb):
        curr_block = i + 1
        feat = RRDB(channels=channels, growth_channels=32, name=f'rrdb_{curr_block}')(feat)

        if curr_block in [4, 9]:
            swin_idx += 1
            feat = SwinBlock(window_size=window_size, num_heads=num_heads,
                             channels=channels, use_shift=False,
                             name=f'swin_w_{swin_idx}')(feat)
            feat = SwinBlock(window_size=window_size, num_heads=num_heads,
                             channels=channels, use_shift=True,
                             name=f'swin_sw_{swin_idx}')(feat)

        if curr_block in [6, 7]:
            mamba_idx += 1
            feat = TFSS2DMambaBlock(dim=channels,
                                    expansion=2.25,
                                    state_dim=8,
                                    res_scale=0.24,
                                    name=f'mamba_core_{mamba_idx}')(feat)

    feat = layers.Conv2D(channels, 3, padding='same',
                         kernel_initializer='he_normal', name='trunk_conv')(feat)
    feat = layers.Add(name='long_skip')([feat_init, feat])

    up = feat
    if scale == 4:
        up = upsample_block(up, channels, scale=2)
        up = upsample_block(up, channels, scale=2)
    else:
        up = upsample_block(up, channels, scale=2)

    outputs = layers.Conv2D(3, 3, padding='same',
                            kernel_initializer='he_normal',
                            activation=None,
                            dtype='float32',
                            name='output_conv')(up)

    return Model(inputs=inputs, outputs=outputs, name='SR_12RRDB_3SwinPairs_Fixed')



# --- 2. Processing Functions ---

# NEW: padding multiple. The Swin window_size is 8, so the input must be a
# multiple of 8 in both H and W. We reflect-pad to the nearest multiple,
# run inference, then crop the SR output back to (orig_H*scale, orig_W*scale).
PAD_MULTIPLE = 8

def _next_multiple(n, m):
    """Smallest multiple of m that is >= n."""
    return ((n + m - 1) // m) * m

def preprocess_input(image_path, pad_multiple=PAD_MULTIPLE):
    """
    Load image, normalize to [-1, 1], and reflection-pad the bottom/right
    so H and W are multiples of `pad_multiple`.
    Returns (batched_tensor, original_H, original_W).
    """
    img = Image.open(image_path).convert('RGB')
    img_arr = np.array(img).astype(np.float32) / 127.5 - 1.0  # (H, W, 3)

    orig_h, orig_w = img_arr.shape[0], img_arr.shape[1]
    target_h = _next_multiple(orig_h, pad_multiple)
    target_w = _next_multiple(orig_w, pad_multiple)

    pad_h = target_h - orig_h
    pad_w = target_w - orig_w

    if pad_h > 0 or pad_w > 0:
        # Reflection padding looks much better at edges than zero padding.
        # 'reflect' mode requires the pad amount to be < the corresponding dim;
        # for tiny images we fall back to 'edge' replication.
        mode = 'reflect' if (pad_h < orig_h and pad_w < orig_w) else 'edge'
        img_arr = np.pad(
            img_arr,
            ((0, pad_h), (0, pad_w), (0, 0)),
            mode=mode
        )

    return np.expand_dims(img_arr, axis=0), orig_h, orig_w

def save_output(img_tensor, output_path, orig_h, orig_w, scale):
    """
    Crop padded SR output back to (orig_h*scale, orig_w*scale),
    denormalize, and save.
    """
    img_arr = img_tensor[0]  # remove batch

    target_h = orig_h * scale
    target_w = orig_w * scale
    # Guard against any rounding mismatches
    target_h = min(target_h, img_arr.shape[0])
    target_w = min(target_w, img_arr.shape[1])
    img_arr = img_arr[:target_h, :target_w, :]

    img_arr = (img_arr + 1.0) * 127.5
    img_arr = np.clip(img_arr, 0, 255).astype(np.uint8)

    Image.fromarray(img_arr).save(output_path)

# --- 3. Main Execution ---
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python enhancer.py <input_path> <output_path> <weights_path>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    weights_path = sys.argv[3]

    try:
        SCALE = 4  # keep your original choice; only change here if you switch weights
        model = build_ultimate_generator1(
                input_shape=(None, None, 3),
                num_rrdb=12, channels=64, scale=SCALE,
                window_size=8, num_heads=2
            )

        if os.path.exists(weights_path):
            model.load_weights(weights_path)
        else:
            print(f"Error: Weights not found at {weights_path}")
            sys.exit(1)

        # Inference with reflection padding
        lr_input, orig_h, orig_w = preprocess_input(input_path, pad_multiple=8)
        sr_output = model.predict(lr_input)

        # Crop SR output to exact (orig*scale) size, then save
        save_output(sr_output, output_path, orig_h, orig_w, SCALE)
        print("Success")

    except Exception:
        traceback.print_exc()
        sys.exit(1)
