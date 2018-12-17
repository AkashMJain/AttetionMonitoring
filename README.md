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
python data_extraction.py -p shape_predictor_68_face_landmarks.dat
```

