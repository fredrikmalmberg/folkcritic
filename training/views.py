from django.shortcuts import render
import importlib
import numpy as np
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

    # If this is a POST request then process the Form data
    if request.method == 'POST':
        post = request.POST
        print('should retrain now with', post['abc'])



    abc_tune, prediction = generate_single_tune(critic)
    print(abc_tune[0][3].strip(' '))
    print(prediction)

    debug_string += 'prediction: ' + str(prediction)
    context = {
        'tune1': abc_tune[0][0],
        'tune2': abc_tune[0][1],
        'tune3': abc_tune[0][2],
        'tune4': abc_tune[0][3].replace(' ', ''),
        'debug_string': debug_string,
    }

    # Render the HTML template index.html with the data in the context variable
    return render(request, 'index.html', context=context)
