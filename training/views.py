from django.shortcuts import render
import importlib
import numpy as np
from training.models import Datapoint, Session, EvaluationResult
from tensorflow import keras


# Tunable params
training_params = {
    "Learning Rate": 0.001,
    "Epochs": 1,
}

from django.contrib.auth.models import User
fg = importlib.import_module('folkcritic-generator')
fc = importlib.import_module('folkcritic')

generator = fg.Generator()
generator.load_pretrained_generator('metadata/folkrnn_v2.pkl')

critic = fc.Critic()
critic.load_model('saved_models/pretrained_model')
critic.recompile()  # this is only done to change metrics for the pretrained model

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
            print('Critic score for generated tune: ' + str(prediction))
            if prediction > 0.5:
                print('breaking')
                break
    return abc_tune, prediction

def get_tunes_for_eval(critic, no_of_tunes = 20, score = 0.5):
    tunes = []
    while len(tunes) < no_of_tunes:
        abc_tune, _, _ = generator.generate_tunes(1, rng_seed=None)
        tune = ' '.join([abc_tune[0][1], abc_tune[0][2], abc_tune[0][3]])
        _ = generator.set_state_from_seed(tune)
        hidden_state = np.expand_dims(np.concatenate(generator.htm1).ravel(), axis=0)
        prediction = critic.predict(hidden_state)[0]
        print('Critic score for generated tune: ' + str(prediction))
        if prediction > score:
            tunes.append(abc_tune)
    return tunes


def eval(request):
    db_session = Session.objects.get(pk=request.session['session'])
    if request.method == 'POST':
        post = request.POST

        # first time we are creating all the tunes for eval
        # and storing them in the user session
        if 'start_eval' in post.keys():
            request.session['total_eval_tunes'] = int(2*(int(post['no_tunes'])//2))
            print('will create this tunes' + str(request.session['total_eval_tunes']))
            no_of_tunes = request.session['total_eval_tunes']/2
            score = 0.5  # This sets the threshold for the tunes selected by the two critics
            request.session['eval_good'] = 0
            request.session['eval_bad'] = 0
            print('starting evaluation')
            try:
                print('Generating tunes with trained critic')
                good_tunes = get_tunes_for_eval(critic, no_of_tunes=no_of_tunes, score=score)
                new_critic = fc.Critic()
                new_critic.load_model('saved_models/pretrained_model')
                new_critic.recompile()
                print('Generating tunes with untrained critic')
                bad_tunes = get_tunes_for_eval(new_critic, no_of_tunes=no_of_tunes, score=score)
                request.session['good_tunes'] = good_tunes
                request.session['bad_tunes'] = bad_tunes
            except:
               print('Creating tunes did not work')
        else:
        # We store the post result and whether the tune came from trained or untrained critic
            if post['good_tune'] == 'True':
                if post['liked'] == 'True':
                    print('liked good tune')
                    request.session['eval_good'] += 1
                else:
                    print('disliked good tune')
            else:
                if post['liked'] == 'True':
                    print('liked bad tune')
                    request.session['eval_bad'] += 1
                else:
                    print('disliked bad tune')

    good_tunes = request.session['good_tunes']
    bad_tunes = request.session['bad_tunes']
    tunes_left = len(good_tunes) + len(bad_tunes)
    total_tunes = request.session['total_eval_tunes']
    test = np.random.rand()
    print(test)
    if tunes_left == 0:
        print('no more tunes to display')
        # Should evaluate here...
        eval_result = 'You liked ' + str(request.session['eval_good']) + ' tunes from the trained critic and ' + \
                      str(request.session['eval_bad']) + ' from the untrained critic (out of ' + str(total_tunes//2) + \
                      ') songs from each.'


        # Saving result to db
        total_tunes = request.session['total_eval_tunes']
        liked_tunes_from_trained = request.session['eval_good']
        liked_tunes_from_untrained = request.session['eval_bad']
        fraction_liked_from_trained = 2*request.session['eval_good']/total_tunes
        fraction_liked_from_untrained = 2*request.session['eval_bad']/total_tunes

        eval_result_db = EvaluationResult(session=db_session, total_tunes = total_tunes, liked_tunes_from_trained = liked_tunes_from_trained,
                                          liked_tunes_from_untrained = liked_tunes_from_untrained,
                                          fraction_liked_from_trained = fraction_liked_from_trained,
                                          fraction_liked_from_untrained = fraction_liked_from_untrained)
        eval_result_db.save()

        context = {
            'eval_result': eval_result,
            'debug_string': '',
            'prediction_string': '',
            'session': db_session,
            'lr': request.session['training_params']['Learning Rate'],
            'epochs': request.session['training_params']['Epochs'],
            'num_datapoints': 'NA',
            'evaluation': True,
            'total_tunes': total_tunes,
            'tunes_left': tunes_left,
        }
        return render(request, 'index.html', context=context)
    else:
        if test < 0.5 and len(good_tunes) > 0:
            abc_tune = good_tunes[0]
            request.session['good_tunes'] = good_tunes[1:]
            good_tune = True
        elif len(bad_tunes) > 0:
            abc_tune = bad_tunes[0]
            request.session['bad_tunes'] = bad_tunes[1:]
            good_tune = False
        else:
            abc_tune = good_tunes[0]
            request.session['good_tunes'] = good_tunes[1:]
            good_tune = True






    context = {
        'tune1': abc_tune[0][0],
        'tune2': abc_tune[0][1],
        'tune3': abc_tune[0][2],
        'tune4': abc_tune[0][3].replace(' ', ''),
        #'retrain_data': retrain_tune,
        'debug_string': '',
        'prediction_string': '',
        'session': db_session,
        'lr': request.session['training_params']['Learning Rate'],
        'epochs': request.session['training_params']['Epochs'],
        'num_datapoints': 'NA',
        'evaluation': True,
        'eval_result': None,
        'total_tunes': total_tunes,
        'tunes_left': tunes_left,
        'good_tune': good_tune,
    }
    return render(request, 'index.html', context=context)

def index(request):
    debug_string = ''
    if 'training_params' not in request.session.keys():
        request.session['training_params'] = training_params

    if request.user.is_authenticated:
        if 'session' not in request.session.keys():
            request.session['session'] = None
        if request.method == 'POST':
            post = request.POST

            # Resetting critic
            if 'reset' in post.keys():
                try:
                    critic.load_model('saved_models/pretrained_model')
                    request.session['training_params'] = training_params
                    critic.recompile(lr=request.session['training_params']['Learning Rate'])
                    print('Reset the critic to pretrained state and reset training params')
                except:
                    print('Failed resetting the critic')

            # Changing training parameters
            if 'lr' in post.keys():
                try:
                    new_training_params = {
                        "Learning Rate": float(post['lr']),
                        "Epochs": int(post['epochs']),
                    }
                    request.session['training_params'] = new_training_params
                    critic.recompile(lr=new_training_params['Learning Rate'])
                    print('Reset training params with: ')
                    for key in request.session['training_params']:
                        print(key + ' now has value ' + str(request.session['training_params'][key]))
                except:
                    print('Failed setting new training parameters')

            # Setting a session if we get one from the picker
            if 'session' in post.keys():
                if post['session'] == 'None':
                    request.session['session'] = None
                else:
                    db_session = Session.objects.get(pk=post['session'])
                    request.session['session'] = post['session']
                    try:
                        load_path = 'saved_models/session_' + str(db_session.name) + '_' + str(db_session.pk) + '/'
                        critic.load_model(load_path)
                        print('Loaded model from session')
                    except:
                        print('Model could not be loaded . Using the basic pretrained one')

            if 'new_session' in post.keys():
                if not Session.objects.filter(name=post['new_session']).exists():
                    db_session = Session(user=request.user, name=post['new_session'])
                    db_session.save()
                    critic.load_model('saved_models/pretrained_model')
                    print('Created a new session and loaded pretrained critic')
                else:
                    # if we have a session with that name we just continue with it and load the critic
                    # This only happens if the name is reused
                    db_session = Session.objects.get(name=post['new_session'])
                    try:
                        load_path = 'saved_models/session_' + str(db_session.name) + '_' + str(db_session.pk) + '/'
                        critic.load_model(load_path)
                        print('Loaded model from session since name is allready in use')
                    except:
                        print('Model could not be loaded . Using the basic pretrained one')
                request.session['session'] = db_session.pk

        if request.session['session'] is None:
            # When no training session is set in the browser session we show the session picker
            available_sessions = Session.objects.filter(user=request.user)
            session_info = []
            for s in available_sessions:
                if not EvaluationResult.objects.filter(session=s).exists():
                    session_info.append([s.pk, s.name, Datapoint.objects.filter(session=s).count(), None, None, None])
                else:
                    print('found several evaluations.. ' + str(EvaluationResult.objects.filter(session=s).count()))
                    e = EvaluationResult.objects.filter(session=s).order_by('-id')[0]
                    session_info.append([s.pk, s.name, Datapoint.objects.filter(session=s).count(), e.total_tunes, 100*e.fraction_liked_from_trained, 100*e.fraction_liked_from_untrained])
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
                        print('Training with ' + str(y.shape[0]) + ' datapoints')
                        critic.train(x, y, epochs=request.session['training_params']['Epochs'])
                        try:
                            save_path = 'saved_models/session_' + str(db_session.name) +'_'+ str(db_session.pk) + '/'
                            critic.model.save(save_path)
                            print('Saved model')
                        except:
                            print('Could not save model')

                    debug_string = 'Retrained critic based on ' + str(data_in_session) + ' tunes.'

            # Otherwise we just show a new tune and no training is done
            abc_tune, prediction = generate_single_tune(critic)
            # retrain_tune is what is posted after the user decides if it is a like or not
            retrain_tune = ' '.join([abc_tune[0][1], abc_tune[0][2], abc_tune[0][3]])

            prediction_string = 'Critic score for the current tune: ' + str(prediction[0])
            context = {
                'tune1': abc_tune[0][0],
                'tune2': abc_tune[0][1],
                'tune3': abc_tune[0][2],
                'tune4': abc_tune[0][3].replace(' ', ''),
                'retrain_data': retrain_tune,
                'debug_string': debug_string,
                'prediction_string': prediction_string,
                'session': db_session,
                'lr': request.session['training_params']['Learning Rate'],
                'epochs': request.session['training_params']['Epochs'],
                'num_datapoints': data_in_session,
                'evaluation': False,
            }
            # Render the HTML template index.html with the data in the context variable
            return render(request, 'index.html', context=context)
    context = {
    }
    return render(request, 'index.html', context=context)
