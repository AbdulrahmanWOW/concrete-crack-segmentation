"""Image loading, dev/test splitting and controlled degradation.

Owner: Data and Baseline Engineer.

Every stage of the project loads images only through this module, so data type,
intensity range and size are identical everywhere:

- ``load_image``: grayscale ``uint8`` 0-255, resized with ``cv2.INTER_AREA``.
- ``load_mask``: binary ``uint8`` 0/1, resized with ``cv2.INTER_NEAREST``.
- ``make_split``: seeded 70 / 30 development / test split written to ``data/splits/``.
"""
