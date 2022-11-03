# Setting up a python virtual environment
Virtual Environments give you to isolate your python packages and dependencies. In python we can create a virtual environment with the built-in venv module.

To create a new venv run the command
```
$ python3 -m venv cs5340-env
```

This will create a new directory called `cs5340-env` which contains your new environment.
Now we can install any python libraries and dependencies we want to use for our project into that environment. But first you need to tell our shell that we want to use this new environment instead of the default python installation. This is called 'activating' the venv. We activate the environment with the command 

```
$ source cs5340-env/bin/activate
```

Once you've activated the environment you should now see the name of the environment in parentheses at the beginning of your command prompt. e.g. `(cs5340-env) [u0000001@lab1-4 ~]$`.

If you run
```
$ which python3
```
You'll also now see that it outputs the path to the python3 inside the environment you have created.

Make sure to activate this environment at the beginning of any session otherwise any python commands you run will use the default python installation instead of your environment.

## Installing a package in your venv
Let's say we would like to use a tokenizer and part of speech tagger for our project from the spacy library. We can install spacy into our venv with pip just as we normally would. (https://spacy.io/usage).

```
$ pip install -U pip setuptools wheel
$ pip install -U spacy
$ python -m spacy download en_core_web_sm
```

These commands make sure that pip setuptools and wheel are updated, installs spacy, and then downloads a spacy model

We've provided a simple test program `pos-tag.py` that you can run to verify that spacy is now installed. You can run the program with

```
$ python3 pos-tag.py
```

If the installation worked then you'll see the part of speech tags for an example sentence printed to stdout.

## Exiting your venv
If you ever want to exit your virtual environment simply run
```
$ deactivate
```
The environment indicator at the prompt will be removed and you'll be back in the original environment.

## Summary
And that's it! The example above showed how to install spacy, but this applies to any python package that you would like to pip install. This is just one way to manage a virtual environment so feel free to use whatever tools are easiest for you. However, we highly encourage you use some form of virtual environment as this will ensure that running your project code is as straightforward and painless as possible!

More info about Python venvs can be found here https://realpython.com/python-virtual-environments-a-primer/ or in python documentation.


# Example of Environment for the BERT Language Model

This is an extra section to show you how to use a virtual environment to load the approved pre-trained language model for the project.

NOTE: You are by no means required to use the language model at all. This is just to help those who would like to try it.

## Installing torch and transformers 
We will loaded the pre-trained bert model using a popular library called HuggingFace Transformers. First you'll need to install this library into a python virtual environment.
The commands below will install specific versions of transformers and a library for working with tensors called torch. Make sure your venv is activated and then install the requirements. 

```
$ source cs5340-env/bin/activate
$ pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
$ pip install transformers==4.7
```

Now you should be ready to go!

## Loading the model
With your venv activated you should now be able to load the model. The model you are allowed to use for this project is called 'bert-base-uncased'. This is a pre-trained version of bert that has NOT been fine-tuned.
You are NOT allowed to use any other language model from huggingface for this project.

The basic way to load this model using huggingface transformers is

```
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained("bert-base-uncased")
```

We have also provided a small script that loads the BERT model, encodes an example sentence and applies the model to it. This is just so you can see that your model has been loaded successfully. Feel free to try it out with.

```
$ python3 bert.py
```
If you see a large tensor printed, then you know it worked.

These links may be helpful when working with the transformers library.
HuggingFace Transformers API docs - https://huggingface.co/docs/transformers/index
The documentation for the specific model - https://huggingface.co/bert-base-uncased?

Good Luck!
