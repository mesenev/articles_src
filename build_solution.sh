docker run --rm \
--env PYTHONPATH=/source \
--workdir /source/phd_3/experiments \
-it -v $(pwd):/source quay.io/fenicsproject/stable:current \
"python3 draw_top_colored.py"
