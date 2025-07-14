# Data Analysis VM

The VM is located at ip `131.215.228.199`. In VSCode, press `Command + Shift + P` (`ctrl + shift + p` on Windows) and type `Remote-SSH: Add New SSH Host`. Here, enter

```
ssh uroc-data@131.215.228.199
```

Then, hit `Enter`. After the window disappears, press `Command + Shift + P` (`ctrl + shift + p` on Windows) and type `Remote-SSH: Connect to Host...`. Here, enter `131.215.228.199` (it should also be the first option). Then, hit enter. You should get a dropdown confirming whether you want to connect to the host, and click yes or whatever other option affirms it. Then, enter the password when prompted.

## Adding files to the VM

In the VM, there is a directory `Documents/data`, which contains two subfolders `optical` and `ultrasound`. One option you have is to plug in a hard drive and `mv` the files to this folder. 

Another option is to `scp` the files to the VM.

For example, if you wanted to upload the directory `temp`, you could do

```
scp -r temp uroc-data@131.215.228.199:/home/uroc-data/Documents/data/optical
```

The `-r` flag works for directories, while without it, it works for files. The path `uroc-data@131.215.228.199:/home/uroc-data/Documents/data/optical` is where the files will be uploaded to on the remote. 