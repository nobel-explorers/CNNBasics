import CNNModules
import random
from IPython.display import clear_output
import ipywidgets as widgets

class Model:
    def __init__(self):
        self.data_loc=""
        self.img_cat=""
        self.X=[]
        self.y=[]
        self.model=None

def all_widgets():

    to_return=[]
    m1=Model()

    to_return.append(widgets.Label("Please enter the url and foldername for the dataset"))
    url=widgets.Text()
    to_return.append(url)

    foldertitle=widgets.Text()
    to_return.append(foldertitle)

    dl_button = widgets.Button(description="Download Images")
    to_return.append(dl_button)

    dl_output = widgets.Output()
    to_return.append(dl_output)

    folder_out=widgets.Label()
    to_return.append(folder_out)

    def dl_on_button_click(b):
        with dl_output:
            clear_output()
            m1.img_cat,m1.data_loc=CNNModules.file_setup(url.value,foldertitle.value)
            folder_out.value="All those images have been downloaded to this location: "+str(m1.data_loc)
            dp_button.disabled=False

    dl_button.on_click(dl_on_button_click)


    to_return.append(widgets.Label('Slide the bar to adjust the size we will reduce the image to'))
    imgsize=widgets.IntSlider(min=1, max=100, value=50)
    to_return.append(imgsize)

    dp_button = widgets.Button(description="Prep Data", disabled=True)
    dp_output = widgets.Output()
    to_return.append(dp_button)
    to_return.append(dp_output)


    def dp_on_button_click(b):
        with dp_output:
            clear_output()
            training_data=CNNModules.data_preprocess(m1.data_loc, imgsize.value, m1.img_cat)
            m1.X,m1.y=CNNModules.restructure_data(training_data)
            ri_button.disabled=False
            tm_button.disabled=False

    dp_button.on_click(dp_on_button_click)

    to_return.append(widgets.Label("Press the button to check out what a random image looks like now"))
    ri_button = widgets.Button(description="Check it out", disabled=True)
    ri_output = widgets.Output()
    matrix_rep=widgets.Label()

    def ri_on_button_click(b):
        with ri_output:
            clear_output()
            randint=random.randint(0,len(m1.X)-1)
            CNNModules.display_image(m1.X[randint])
            matrix_rep.value=m1.X[randint]

    ri_button.on_click(ri_on_button_click)

    to_return.append(ri_button)
    to_return.append(ri_output)
    to_return.append(matrix_rep)

    to_return.append(widgets.Label("Adjust the following variables and click the button below to train your model on the given dataset"))

    to_return.append(widgets.Label("Number of Hidden Layers"))
    numlayers = widgets.widgets.BoundedIntText(value=1,min=0,max=4,step=1,disabled=False)
    to_return.append(numlayers)

    to_return.append(widgets.Label("Number of Nodes per Layer"))
    numnodes = widgets.widgets.BoundedIntText(value=64,min=1,max=100,step=1,disabled=False)
    to_return.append(numnodes)

    to_return.append(widgets.Label("Batch Size"))
    batchsize = widgets.widgets.BoundedIntText(value=32,min=1,max=200,step=1,disabled=False)
    to_return.append(batchsize)

    to_return.append(widgets.Label("Epochs"))
    epochs = widgets.widgets.BoundedIntText(value=3,min=1,max=10,step=1,disabled=False)
    to_return.append(epochs)

    tm_button = widgets.Button(description="Train the model",disabled=True)
    to_return.append(tm_button)

    tm_output = widgets.Output()
    to_return.append(tm_output)

    def tm_on_button_clicked(b):
        with tm_output:
            clear_output()
            model, model_metrics=CNNModules.build_model(m1.X, m1.y, numlayers.value, numnodes.value, len(m1.img_cat), batchsize.value, epochs.value)
            m1.model=model

    tm_button.on_click(tm_on_button_clicked)

    return to_return







