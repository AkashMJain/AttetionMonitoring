# AttetionMonitoring
## Dependence

scipy

  `sudo pip install scipy`

python-opencv

  `sudo pip install python-opencv`

imutils

  `sudo pip install imutils`

dlib

  `sudo pip install dlib`

  OR

  ```
  step 1

    download file
      https://pypi.org/project/dlib/#files

  step 2

    extract above file by executing command
    $ tar -xvf 'filename.tar.gz'

  step 3

    cd to dlib extracted folder and run following command.
    $ python setup.py install

  ```


## Install

```
git clone git@github.com:AkashMJain/AttetionMonitoring.git

cd AttetionMonitoring

python AttentionMonitoring.py -p caffe_files/deploy.prototxt.txt -m caffe_files/res10_300x300_ssd_iter_140000.caffemode -c "(camera ID in Integers (default is 0))"
```

