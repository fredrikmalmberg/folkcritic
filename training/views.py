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

def retrain(critic):

    return critic

def index(request):
    debug_string = ''

    if request.user.is_authenticated:
        if 'session' not in request.session.keys():
            request.session['session'] = None
        if request.method == 'POST':
            post = request.POST
            # Setting a session if we get one from the picker
            if 'session' in post.keys():
                if post['session'] == 'None':
                    request.session['session'] = None
                else:
                    request.session['session'] = post['session']
            if 'new_session' in post.keys():
                if not Session.objects.filter(name=post['new_session']).exists():
                    session = Session(user=request.user, name=post['new_session'])
                    session.save()
                else:
                    # if we have a session with that name we just continue with it
                    session = Session.objects.get(name=post['new_session'])
                request.session['session'] = session.pk

        if request.session['session'] is None:
            # When no session is yet selected we show the session picker
            available_sessions = Session.objects.filter(user=request.user)
            session_info = []
            for s in available_sessions:
                session_info.append([s.pk, s.name, Datapoint.objects.filter(session=s).count()])
            context = {
                'session': None,
                'available_sessions': session_info
            }
            return render(request, 'index.html', context=context)
        else:
            # Case when there is a session selected, continue training
            db_session = Session.objects.get(pk=request.session['session'])
            data_in_session = len(Datapoint.objects.filter(session=db_session))
            # Save and retrain if we get data in post.
            if request.method == 'POST':
                post = request.POST
                if 'retrain_data' in post.keys():
                    if not Datapoint.objects.filter(tune=post['retrain_data']).exists():
                        data = Datapoint(session=db_session, tune=post['retrain_data'], liked=post['liked'])
                        data.save()
                        data_in_session += 1
                        # Should retrain here
                        all_data = Datapoint.objects.filter(session=db_session)
                        x = None
                        y = None
                        for idx, d in enumerate(all_data):
                            _ = generator.set_state_from_seed(d.tune)
                            state_to_store = np.expand_dims(np.concatenate(generator.htm1).ravel(), axis=0)
                            target_to_store = np.ones(state_to_store.shape[0]) if d.liked else np.zeros(state_to_store.shape[0])
                            if idx == 0:
                                x = state_to_store
                                y = np.array(target_to_store)
                            else:
                                x = np.append(np.copy(x), state_to_store, axis=0)
                                y = np.append(np.copy(y), target_to_store, axis=0)
                            critic.train(x, y)

                    debug_string = 'Retrained critic based on ' + str(data_in_session) + ' tunes.'

            # Otherwise we just show something new
            # print('should retrain now with', post['retrain_data'], post['liked'])

            abc_tune, prediction = generate_single_tune(critic)
            retrain_tune = ' '.join([abc_tune[0][1], abc_tune[0][2], abc_tune[0][3]])

            prediction_string = 'Score for current tune: ' + str(prediction[0])
            context = {
                'tune1': abc_tune[0][0],
                'tune2': abc_tune[0][1],
                'tune3': abc_tune[0][2],
                'tune4': abc_tune[0][3].replace(' ', ''),
                'retrain_data': retrain_tune,
                'debug_string': debug_string,
                'prediction_string': prediction_string,
                'session': db_session,
                'num_datapoints': data_in_session
            }
            # Render the HTML template index.html with the data in the context variable
            return render(request, 'index.html', context=context)
    context = {
    }
    return render(request, 'index.html', context=context)
