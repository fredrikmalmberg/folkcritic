# folkcritic (WIP)
Building on top of https://github.com/IraKorshunova/folk-rnn/ and introducing a 'critic' that actively learns from feedback provided by a human in the loop  
## if using conda
$ conda install --file requirements.txt
## update database
$ python manage.py makemigrations
$ python manage.py migrate
## create a user
$ python manage.py createsuperuser
## Running the server
$ python manage.py runserver
