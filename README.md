# Sign-Language-To-Text-Conversion

## Abstract

Sign language is one of the oldest and most natural form of language for communication, but since most people do not know sign language and interpreters are very difficult to come by we have come up with a real time method using neural networks for fingerspelling based american sign language.<br> 
In this method, the hand is first passed through a filter and after the filter is applied the hand is passed through a classifier which predicts the class of the hand gestures. This method provides 95.7 % accuracy for the 26 letters of the alphabet.

![](signs.png)


## Libraries Required for this project  -(Requires the latest pip version to install all the packages)
Note : Python 3.6 is required to build this project, as some of the libraries required can't be installed on the lastest version of the Python 

```
1. Lastest pip -> pip install --upgrade pip
2. numpy -> pip install numpy
3. string -> pip install strings
4. os-sys -> pip install os-sys
5. opencv -> pip install opencv-python
6. tensorFlow -> i) pip install tensorflow <br>
                 ii) pip3 install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.8.0-cp34-cp34m-linux_x86_64.whl<br>
7. keras -> pip install keras
8. tkinter -> pip install tk
9. PIL -> pip install Pillow
10. enchant -> pip install pyenchant (Python bindings for the Enchant spellchecking system)
11. hunspell -> pip install cyhunspell (A wrapper on hunspell for use in Python)
```

### Running the Project 

```
python /path/to/the/Application.py
```

### License
Copyright (c) 2021 Nikhil Gupta

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
