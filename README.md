![mi](https://github.com/SivadithiyanOfcl/EDM-MGC/assets/113179307/d41e8014-936a-44a0-a886-c3f916a88e3c)# EDM-MGC


🎵 EDM-MGC (Electronic Dance Music Music Genre Classification) is a project aimed at automatically categorizing music tracks into different EDM sub-genres. By leveraging machine ML algorithms, it analyzes audio features to identify characteristics unique to each genre, such as MFCC, Chroma, and Spectral features. With EDM-MGC, users can efficiently organize and discover EDM tracks, enhancing their music listening experience! 🎧💃🔊

## Output

![cremate](https://github.com/SivadithiyanOfcl/EDM-MGC/assets/113179307/e105b5f3-38e9-4068-9b7e-fa33af1b737e)
![Hope](https://github.com/SivadithiyanOfcl/EDM-MGC/assets/113179307/184ddf35-98d5-483c-83c9-897077ebb3af)
![opp](https://github.com/SivadithiyanOfcl/EDM-MGC/assets/113179307/b7a60d98-e34f-441c-950a-8ab7019226e2)
![mi](https://github.com/SivadithiyanOfcl/EDM-MGC/assets/113179307/bca88fb0-0163-4ffc-87cf-72c73ba6a62c)


https://github.com/SivadithiyanOfcl/EDM-MGC/assets/113179307/f89b19e9-d8a7-4250-bb19-99722c749b07



## Setup
To get this repository, run the following command inside your git-enabled terminal:

```git clone https://github.com/SivadithiyanOfcl/EDM-MGC.git```

Note: In-order to run the project, use your base/virtual environment. and locate to the requirements.txt in your folder and install the necessary libraries.

```pip install -r requirements.txt```

You need to make one last adjustment before running the code:
Locate the utils.py file under the EDMMGC\myapp and replace the following line of code

```BASE_DIR = r'D:\path\to\folder\EDMMGC'``` to where ever this project was located in your local device.

Once you have downloaded Django, navigate to the cloned repo directory and run the following:

```
python manage.py makemigrations
python manage.py migrate
python manage.py createsuperusere
python manage.py runserver
```

Once the server is hosted, visit [http://127.0.0.1:8000/](http://127.0.0.1:8000/) to access the app.

