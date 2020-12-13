from django.shortcuts import render
import importlib
import numpy as np
from training.models import Datapoint, Session
from django.contrib.auth.models import User
fg = importlib.import_module('folkcritic-generator')
fc = importlib.import_module('folkcritic')

generator = fg.Generator()
generator.load_pretrained_generator('metadata/folkrnn_v2.pkl')

critic = fc.Critic()
critic.load_model('saved_models/')

#data_path = 'state_data/'
#x_real, x_generated, real_tunes, generated_tunes = critic.preprocess_data(data_path)

def generate_single_tune(critic):
    prediction = 0
    abc_tune = None
    batch = 1
    while prediction < 0.5:
        abc_tunes, _, _ = generator.generate_tunes(batch, rng_seed=None)
        for i in range(batch):
            abc_tune = abc_tunes[i:i+1]
            tune = ' '.join([abc_tune[0][1], abc_tune[0][2], abc_tune[0][3]])
            _ = generator.set_state_from_seed(tune)
            hidden_state = np.expand_dims(np.concatenate(generator.htm1).ravel(), axis=0)
            prediction = critic.predict(hidden_state)[0]
            if prediction > 0.5:
                break
    return abc_tune, prediction

def index(request):
    debug_string = ''
    #if request.user.is_authenticated:
        #if request.user.pk
        #session = Session(user=request.user, name='1')
        #print('user logged in and creating session')
        #print(request.user.pk)
    #else:
    #    session = Session(None, name='test')
    #session.save()
    # If this is a POST request then process the Form data

    if request.user.is_authenticated:
        if request.method == 'POST':
            post = request.POST
            if 'session' in post.keys():
                session = Session.objects.get(pk=post['session'])
                if len(Datapoint.objects.filter(session=session)) == 0:
                    print("no datapoints in this session")
                else:
                    print("datapoints in this session!", post['session'])
                # Save and retrain here if we get data in post...
                if 'retrain_data' in post.keys():
                    data = Datapoint(session=session, tune=post['retrain_data'], liked=post['liked'])
                    data.save()
                    # Should retrain here
                # Otherwise we just show something new
                #print('should retrain now with', post['retrain_data'], post['liked'])
                #_ = generator.set_state_from_seed(post['retrain_data'])
                #hidden = np.expand_dims(np.concatenate(generator.htm1).ravel(), axis=0)

                #critic.train_single(hidden)
                #debug_string = 'Retrained critic based on feedback'
                abc_tune, prediction = generate_single_tune(critic)
                print(abc_tune[0])
                print(prediction)
                retrain_tune = ' '.join([abc_tune[0][1], abc_tune[0][2], abc_tune[0][3]])

                prediction_string = ' prediction: ' + str(prediction)
                context = {
                    'tune1': abc_tune[0][0],
                    'tune2': abc_tune[0][1],
                    'tune3': abc_tune[0][2],
                    'tune4': abc_tune[0][3].replace(' ', ''),
                    'retrain_data': retrain_tune,
                    'debug_string': debug_string,
                    'prediction_string': prediction_string,
                    'session': session
                }

                # Render the HTML template index.html with the data in the context variable
                return render(request, 'index.html', context=context)

        available_sessions = Session.objects.filter(user = request.user)
        for ses in available_sessions:
            print(ses.pk)
        context = {
            'session': None,
            'available_sessions': available_sessions
        }
    else:
        context = {
            'session': None
        }
    return render(request, 'index.html', context=context)
