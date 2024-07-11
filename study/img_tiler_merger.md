## Tiling & Merging

- https://github.com/the-lay/tiler

```python
def show_tiles(tiles, nrows, ncols, tile_shape):
    tile_h, tile_w = tile_shape
    fig, axes = plt.subplots(nrows, ncols, figsize=(8, 8))
    for i in range(nrows):
        for j in range(ncols):
            axes[i][j].imshow(tiles[ncols*i + j])
            axes[i][j].set_xlim(0, tile_w); axes[i][j].set_xticks([0, tile_w]); 
            axes[i][j].set_ylim(tile_h, 0); axes[i][j].set_yticks([0, tile_h]); 

    fig.tight_layout()
    plt.show()
```

### No Padding

```python
## No padding
h, w, n_channels = 500, 700, 3
tile_h, tile_w = 200, 200

img = skimage.data.astronaut()
img = skimage.transform.resize(img, (h, w))

tiler = Tiler(data_shape=img.shape, channel_dimension=-1,
              tile_shape=(tile_h, tile_w, n_channels), )
print(tiler)

merger = Merger(tiler=tiler)
tiles = []
for tile_id, tile in tiler(img):
    tiles.append(tile)
    merger.add(tile_id, tile)

nrows, ncols = tiler._indexing_shape[:2]
show_tiles(tiles, nrows, ncols, tile_shape=(tile_h, tile_w))

img_merged = merger.merge(unpad=True)  # (1920, 1080, 3)
print(img.shape, img_merged.shape)

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 8))
ax1.imshow(img)
ax2.imshow(img_merged)
fig.tight_layout()
plt.show()
```

### Padding (Overlap)

```python
## Padding with overlap

h, w, n_channels = 500, 700, 3
tile_h, tile_w = 200, 200
overlap = 20

img = skimage.data.astronaut()
img = skimage.transform.resize(img, (h, w))

tiler = Tiler(data_shape=img.shape, channel_dimension=-1,
              tile_shape=(tile_h + overlap, tile_w + overlap, n_channels),
              overlap=overlap,
              mode="drop",)
# print(tiler)

new_shape, padding = tiler.calculate_padding()
tiler.recalculate(data_shape=new_shape)
padded_img = np.pad(img, padding, mode="reflect")
# print(tiler)

merger = Merger(tiler=tiler, window="overlap-tile")
tiles = []
for tile_id, tile in tiler(padded_img):
    tiles.append(tile)
    merger.add(tile_id, tile)

nrows, ncols = tiler._indexing_shape[:2]
show_tiles(tiles, nrows, ncols, tile_shape=(tile_h, tile_w))

img_merged = merger.merge(extra_padding=padding, dtype=img.dtype)
print(img.shape, img_merged.shape)

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 8))
ax1.imshow(img)
ax2.imshow(img_merged)
fig.tight_layout()
plt.show()
```
