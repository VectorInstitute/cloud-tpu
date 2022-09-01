# TPU on Google Cloud
A repository containing reference implementations for running various types of models on Google Cloud's TPUs. The notebooks are written in PyTorch Lightning, and run on TPU devices using PyTorch's XLA library. You can learn more about TPU training with PyTorch Lightning [here](https://pytorch-lightning.readthedocs.io/en/stable/notebooks/lightning_examples/mnist-tpu-training.html), and about PyTorch on XLA devices [here](https://pytorch.org/xla/release/1.12/index.html).

# Connecting
Below are the steps to connecting to our TPU VMs on GCP, and getting Jupyter running:

If you are on the Vector cluster, you can't install the GCP CLI tool directly (permission). Download the SDK from source, and place it in your home directory. Then, for any commands starting with `gcloud` below, you can run it in the vector cluster by replacing `gcloud` with the following:
```
~/google-cloud-sdk/bin/gcloud
```
<br>

**upload** <br>
Below is an example demonstrating how to move an example file from the cluster to the shared drive on the TPU VMs using the `scp` command.
```
~/google-cloud-sdk/bin/gcloud alpha compute tpus tpu-vm scp \
/ssd003/projects/aieng/conversational_ai/example \
convai-tpu-vm-1:/shared/example \
--zone=us-central1-b 
--recurse
```
<br>

**describe** <br>
You can use `describe` to display the status and additional information about the TPU VM you would like to use. The most important field to note is the `STATUS`, which will tell you whether your TPU VM is `STOPPED` (meaning you need to start it) or `READY` (meaning you can now ssh to it).
```
gcloud compute tpus tpu-vm describe convai-tpu-vm-1 --zone=us-central1-b
```
<br>

**start** <br>
If your TPU VM status is `STOPPED`, you can start it with this command. **NOTE: Make sure to stop your VM after use, as TPU usage is very expensive!!**
```
gcloud compute tpus tpu-vm start convai-tpu-vm-1 --zone=us-central1-b
```
<br>

Occasionally, the TPU VM will fail to start with the following error:
```
There is no more capacity in the zone \"us-central1-b\"; you can try in another
zone where Cloud TPU Nodes are offered (see
https://cloud.google.com/tpu/docs/regions)
```
Just try again! It will typically work within the next 1-2 minutes.

<br>

**ssh** <br>
Once you've started your TPU VM, you can ssh using the following command:

```
gcloud compute tpus tpu-vm ssh convai-tpu-vm-1 --zone=us-central1-b
```
<br>

**stop** <br>
⚠️ **Always remember to stop your TPU VM after use!! Use the command below.**
```
gcloud compute tpus tpu-vm stop convai-tpu-vm-1 --zone=us-central1-b
```
<br> 


**jupyter** <br>
To start jupyter, type in the following commands once you're connected to the TPU VM through ssh.

```
cd /
jupyter lab --ip `hostname -i` --no-browser
```

Then, open a new shell tab or window, and grab the TPU VM ip address as such:
```
gcloud compute tpus tpu-vm describe convai-tpu-vm-1 --zone=us-central1-b | grep ipAddress
```

In the new window, type the following command (substitute <tpu-vm-ip> with the ip address from the previous command):
```
gcloud compute ssh jupyter-tunnel -- -L 8888:<tpu-vm-ip>:8888
```


# Examples
**Autoencoder** <br>
The simplest possible lightning model from the introduction section to [PyTorch Lightning Docs](https://pytorch-lightning.readthedocs.io/en/stable/starter/introduction.html). The original example is a simple tutorial on PyTorch Lightning alone, and doesn't contain any XLA instructions. I would recommend reading the original tutorial first, and then coming back to this notebook which I've adapted to use XLA. This should give you a thorough understanding of both Lightning and the XLA modifications you need to make to your code in order to get it running on TPU.

**MNIST** <br>
The classic handwritten digit recognition dataset with MNIST, trained on a simple neural network configured from PyTorch Lightning's `LightningModule`. Adapted to use XLA with 8 TPU cores for training.

**NLP** <br>
The large language model RoBERTa trained on a sentiment analysis task, adapted to use XLA with 8 TPU cores for training. This is a bit more involved, so please take a look at the previous two notebooks first!

# Tips

In your Jupyter Notebooks, always define the following two environment variables to configure your XLA device (TPU). The first gives GCP the IP address to TPU worker 0, which contains all of your TPU configuration. The second tells PyTorch to use he bFloat16 format, rather than Float32. This will maximize TPU performance and circumvent memory errors that would pop up otherwise.

```
%env XRT_TPU_CONFIG=localservice;0;localhost:51011
%env XLA_USE_BF16=1
```

# Related Issues
Training on 8 TPU cores, for some reason, only works intermittently on GCP. You can see an example of this error in the autoencoder notebook. An issue has been opened by @markcoatsworth and I, and is being tracked by the folks at Google. You can view it here: https://github.com/pytorch/xla/issues/3947
