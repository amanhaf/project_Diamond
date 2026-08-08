import uuid

# Freeze this forever once you start generating UUIDs in production.
DIAMOND_NAMESPACE = uuid.UUID("2b5c2a8c-8b0c-4b2d-9d1c-3f2f5c2a8c10")

# Controls how tight/loose the visual fingerprint is
PHASH_SIZE = 16   # 8, 16, 32; larger = more detail, slower
CANON_SIZE = 512  # normalize render size for stability
MARGIN_PX = 8     # padding after crop
