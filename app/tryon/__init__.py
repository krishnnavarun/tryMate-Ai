"""Virtual try-on providers (Phase 5).

base.py defines the TryOnProvider interface; replicate_idm.py and mock.py implement it,
so the provider can be swapped (e.g. CatVTON, self-hosted) without changing the API.
"""
