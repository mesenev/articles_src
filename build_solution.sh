docker run --rm \
--env PYTHONPATH=/source:/source/phd_3/experiments \
--name fenics \
-v $(pwd):/source \
--workdir /source/phd_3/experiments \
-p 127.0.0.1:8888:8888 \
-it \
quay.io/fenicsproject/stable:current \
'jupyter-notebook --ip=0.0.0.0 --NotebookApp.token="tokentoken" & /bin/bash'

#--network host \     # gethostbyname failed
