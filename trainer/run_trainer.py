import os
from random import randint

run_name = "for_luck"


if __name__ == "__main__":
    os.chdir("..")
    os.system("gcloud ml-engine jobs submit training \"{0}\" --job-dir gs://pyeye_bucket/jobs --runtime-version 1.4 --module-name trainer.cnn --package-path ./trainer --config trainer/cloudml-gpu.yaml --region us-east1".format(run_name))
