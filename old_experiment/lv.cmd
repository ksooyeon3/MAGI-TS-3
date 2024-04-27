universe = docker
docker_image = shihaoyangphd/py37env:v0402
requirements = (Machine == "isye-syang605.isye.gatech.edu")
executable = /usr/local/bin/python3
arguments = scripts/experiment.py -p params/lv.config -r results/lv/m31/ -s 20221030

should_transfer_files = YES
when_to_transfer_output = ON_EXIT
transfer_input_files = scripts, data, params, results
transfer_output_files = results

error = $(Cluster).$(Process).err
log = $(Cluster).$(Process).log
notification = error
notification = complete
notify_user = chuang397@gatech.edu

request_memory = 2048M

queue
