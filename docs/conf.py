import pituplots

project = "pituPLOTS"
author = "Pablo Iturrieta"
release = pituplots.__version__

extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx_gallery.gen_gallery"]
exclude_patterns = ["_build"]

sphinx_gallery_conf = {
    "examples_dirs": "../examples",
    "gallery_dirs": "gallery",
    "download_all_examples": False,
}
