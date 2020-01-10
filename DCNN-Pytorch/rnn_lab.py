
# coding: utf-8

# # Recurrent Neural Networks
# 
# In this lab we will experiment with recurrent neural networks. We will build a text generation model that predicts a word given the previous words, and hence will allow us to generate a sentence. This can easily be extended to generating a sentence description for a given input image. RNNs are a useful type of model for predicting sequences or handling sequences of things as inputs. In this lab we will use again Pytorch's nn library.
# 
# We will also be using the COCO dataset which includes images + textual descriptions (captions) + other annotations. We can browse the dataset here: http://cocodataset.org/#home
# 
# First, let's import libraries and make sure we have everything properly installed.

# In[1]:


import torch, json, string
import torch.nn as nn
import torch.autograd
from torch.autograd import Variable
import torch.utils.data as data
import numpy as np
from nltk.tokenize import word_tokenize
import nltk
#nltk.download()


# ## 1. Loading and Preprocessing the Text
# Pytorch comes with a Dataset class for the COCO dataset but I will write my own class here. This class does two important things: 1) Building a vocabulary with the most frequent words, 2) Building utilities to convert a sentence into a list of word ids, and back. We are not going to be using the images for the purposes of the lab but you will use them in the assignment questions.

# In[2]:


from tqdm import tqdm_notebook as tqdm

class CocoCaptions(data.Dataset):
    # Load annotations in the initialization of the object.
    def __init__(self, captionsFile, vocabulary = None):
        self.data = json.load(open(captionsFile))
        self.imageIds = self.data['images']
        self.annotations = self.data['annotations']
        
        # Build a vocabulary if not provided.
        if not vocabulary:
            self.build_vocabulary()
        else:
            self.vocabulary = vocabulary
        
    # Build a vocabulary using the top 5000 words.
    def build_vocabulary(self, vocabularySize = 5000):
        # Count words, this will take a while.
        word_counter = dict()
        for annotation in tqdm(self.annotations, desc = 'Building vocabulary'):
            words = word_tokenize(annotation['caption'].lower())
            for word in words:
                word_counter[word] = word_counter.get(word, 0) + 1
                
        # Sort the words and find keep only the most frequent words.
        sorted_words = sorted(list(word_counter.items()), 
                              key = lambda x: -x[1])
        most_frequent_words = [w for (w, c) in sorted_words[:vocabularySize]]
        word2id = {w: (index + 1) for (index, w) in enumerate(most_frequent_words)}
        
        # Add a special characters for START, END sentence, and UNKnown words.
        word2id['[END]'] = 0
        word2id['[START]'] = len(word2id)
        word2id['UNK'] = len(word2id)
        id2word = {index: w for (w, index) in word2id.items()}
        self.vocabulary = {'word2id': word2id, 'id2word': id2word}
    
    # Transform a caption into a list of word ids.
    def caption2ids(self, caption):
        word2id = self.vocabulary['word2id']
        caption_ids = [word2id.get(w, word2id['UNK']) for w in word_tokenize(caption.lower())]
        caption_ids.insert(0, word2id['[START]'])
        caption_ids.append(word2id['[END]'])
        return torch.LongTensor(caption_ids)
    
    # Transform a list of word ids into a caption.
    def ids2caption(self, caption_ids):
        id2word = self.vocabulary['id2word']
        return string.join([id2word[w] for w in caption_ids], " ")
    
    # Return imgId, and a random caption for that image.
    def __getitem__(self, index):
        annotation = self.annotations[index]
        return annotation['image_id'], self.caption2ids(annotation['caption'])
    
    # Return the number of elements of the dataset.
    def __len__(self):
        return len(self.annotations)
    
# Let's test the data class.
trainData = CocoCaptions('annotations/captions_train2014.json')
print('Number of training examples: ', len(trainData))

# It would be a mistake to build a vocabulary using the validation set so we reuse.
valData = CocoCaptions('annotations/captions_val2014.json', vocabulary = trainData.vocabulary)
print('Number of validation examples: ', len(valData))

# Print a sample from the training data.
imgId, caption = trainData[0]
print('imgId', imgId)
print('caption', caption.tolist())
print('captionString', trainData.ids2caption(caption))


# ## 2. Making a Data Loader that can Handle Sequences.
# 
# Handling sequences is special when processing batches of inputs because each sequence can have a different length. This makes batching complicated, and different libraries have different ways of handling this which might be easier or harder to deal with. Here we are padding the sequences to the maximum sequence length in a given batch. Additionally pytorch has nice utility functions that require sorting the sequences in a batch from longest to shortest.

# In[3]:


# The batch builder will pack all sequences of different length into a single tensor by 
# padding shorter sequences with a padding token.
def customBatchBuilder(samples):
    imgIds, captionSeqs = zip(*samples)
    
    # Sort sequences based on length.
    seqLengths = [len(seq) for seq in captionSeqs]
    maxSeqLength = max(seqLengths)
    sorted_list = sorted(zip(list(imgIds), captionSeqs, seqLengths), key = lambda x: -x[2])
    imgIds, captionSeqs, seqLengths = zip(*sorted_list)
    
    # Create tensor with padded sequences.
    paddedSeqs = torch.LongTensor(len(imgIds), maxSeqLength)
    paddedSeqs.fill_(0)
    for (i, seq) in enumerate(captionSeqs):
        paddedSeqs[i, :len(seq)] = seq
    return imgIds, paddedSeqs.t(), seqLengths

# Data loaders in pytorch can use a custom batch builder, which we are using here.
trainLoader = data.DataLoader(trainData, batch_size = 128, 
                              shuffle = True, num_workers = 0,
                              collate_fn = customBatchBuilder)
valLoader = data.DataLoader(valData, batch_size = 128, 
                            shuffle = False, num_workers = 0,
                            collate_fn = customBatchBuilder)

# Now let's try using the data loader.
index, (imgIds, paddedSeqs, seqLengths) = next(enumerate(trainLoader))
print('imgIds', imgIds)
print('paddedSequences', paddedSeqs.size())
print('seqLengths', seqLengths)


# 
# 
# ## 3. Building our model using a Recurrent Neural Network.
# We will build a model that predicts the next word based on the previous word using a recurrent neural network. Additionally we will be using an Embedding layer which will assign a unique vector to each word. The network will be trained with a softmax + negative log likelihood loss. Similar to classification we will be trying to optimize for the correct word at each time-step.

# In[4]:


# By now, we should know that pytorch has a functional implementation (as opposed to class version)
# of many common layers, which is especially useful for layers that do not have any parameters.
# e.g. relu, sigmoid, softmax, etc.
import torch.nn.functional as F

class TextGeneratorModel(nn.Module):
    # The model has three layers: 
    #    1. An Embedding layer that turns a sequence of word ids into 
    #       a sequence of vectors of fixed size: embeddingSize.
    #    2. An RNN layer that turns the sequence of embedding vectors into 
    #       a sequence of hiddenStates.
    #    3. A classification layer that turns a sequence of hidden states into a 
    #       sequence of softmax outputs.
    def __init__(self, vocabularySize):
        super(TextGeneratorModel, self).__init__()
        # See documentation for nn.Embedding here:
        # http://pytorch.org/docs/master/nn.html#torch.nn.Embedding
        self.embedder = nn.Embedding(vocabularySize, 300)
        self.rnn = nn.RNN(300, 512, batch_first = False)
        self.classifier = nn.Linear(512, vocabularySize)
        self.vocabularySize = vocabularySize

    # The forward pass makes the sequences go through the three layers defined above.
    def forward(self, paddedSeqs, initialHiddenState):
        batchSequenceLength = paddedSeqs.size(0)  # 0-dim is sequence-length-dim.
        batchSize = paddedSeqs.size(1)  # 1-dim is batch dimension.
        
        # Transform word ids into an embedding vector.
        embeddingVectors = self.embedder(paddedSeqs)
        
        # Pass the sequence of word embeddings to the RNN.
        rnnOutput, finalHiddenState = self.rnn(embeddingVectors, initialHiddenState)
        
        # Collapse the batch and sequence-length dimensions in order to use nn.Linear.
        flatSeqOutput = rnnOutput.view(-1, 512)
        predictions = self.classifier(flatSeqOutput)
        
        # Expand back the batch and sequence-length dimensions and return. 
        return predictions.view(batchSequenceLength, batchSize, self.vocabularySize),                finalHiddenState

# Let's test the model on some input batch.
vocabularySize = len(trainData.vocabulary['word2id'])
model = TextGeneratorModel(vocabularySize)

# Create the initial hidden state for the RNN.
index, (imgIds, paddedSeqs, seqLengths) = next(enumerate(trainLoader))
initialHiddenState = Variable(torch.Tensor(1, paddedSeqs.size(1), 512).zero_())
predictions, _ = model(torch.autograd.Variable(paddedSeqs), initialHiddenState)

print('Here are input and output size tensor sizes:')
# Inputs are seqLength x batchSize x 1 
print('inputs', paddedSeqs.size()) # 10 input sequences.
# Outputs are seqLength x batchSize x vocabularySize
print('outputs', predictions.size()) # 10 output softmax predictions over our vocabularySize outputs.


# ## 3. Sampling a New Sentence from the Model.
# 
# The code below uses the RNN network as an RNN cell where we only pass one single input word, and a hidden state vector. Then we keep passing the previously predicted word, and previously predicted hidden state to predict the next word. Since the given model is not trained, it will just output a random sequence of words for now. Ideally, the trained model should also learn when to [END] a sentence.

# In[5]:


def sample_sentence(model, use_cuda = False):
    counter = 0
    limit = 200
    words = list()

    # Setup initial input state, and input word (we use "the").
    previousWord = torch.LongTensor(1, 1).fill_(trainData.vocabulary['word2id']['the'])
    previousHiddenState = torch.autograd.Variable(torch.Tensor(1, 1, 512).zero_())
    if use_cuda: previousHiddenState = previousHiddenState.cuda()

    while True:
        # Predict the next word based on the previous hidden state and previous word.
        inputWord = torch.autograd.Variable(previousWord)
        if use_cuda: inputWord = inputWord.cuda()
        predictions, hiddenState = model(inputWord, previousHiddenState)
        nextWordId = np.random.multinomial(1, F.softmax(predictions.squeeze()).data.cpu().numpy(), 1).argmax()
        words.append(trainData.vocabulary['id2word'][nextWordId])
        # Setup the inputs for the next round.
        previousWord.fill_(nextWordId)
        previousHiddenState = hiddenState

        # Keep adding words until the [END] token is generated.
        if nextWordId == trainData.vocabulary['word2id']['[END]'] or counter > limit:
            break
        counter += 1
    
    words.insert(0, 'the')
    words.insert(0, '[START]')
    return string.join(words, " ")

print(sample_sentence(model, use_cuda = False))


# ## 3. Training the Model
# 
# Now that data is pre-processed, we can try training the model. An important part is to define our target labels or ground-truth labels. In this text generation model, we want to predict the next word based on the previous word. So we need to provide as the target a shifted version of the input sequence. The code below looks a lot like the code used for training previous models with only small modifications.

# In[6]:


import tqdm as tqdmx
from tqdm import tqdm_notebook as tqdm
tqdmx.tqdm.get_lock().locks = []
import torch.autograd

def train_rnn_model(network, criterion, optimizer, trainLoader, valLoader, n_epochs = 10, use_gpu = False):
    if use_gpu:
        network = network.cuda()
        criterion = criterion.cuda()
        
    # Training loop.
    for epoch in range(0, n_epochs):
        
        cum_loss = 0.0
        counter = 0.0
        correct = 0.0
        total_words = 0.0   
        # Make a pass over the training data.
        t = tqdm(trainLoader, desc = 'Training epoch %d' % epoch)
        network.train()  # This is important to call before training!
        for (i, (imgIds, paddedSeqs, seqLengths)) in enumerate(t):
         #   print(im)
         #   print(paddedSeqs.size())
         #   print(hidden_state)inputs
            # Wrap inputs, and targets into torch.autograd.Variable types.
            hidden_state = Variable(torch.Tensor(1, paddedSeqs.size(1), 512).zero_())
            #chop off the "END" word for the input (we want to train the network to predict this)
            input_sentence = torch.autograd.Variable(paddedSeqs[:-1])
            #chop off the "START" word the ground-truth
            label_sentence = torch.autograd.Variable(paddedSeqs[1:])
            if use_gpu:
                label_sentence = label_sentence.cuda()
                input_sentence = input_sentence.cuda()
                hidden_state = hidden_state.cuda()

            # Forward pass:
            outputs, hidden_state  = model(input_sentence, hidden_state)
            #collapse sequence length and batch size dimensions and force output
            #to be (seq_length * batch_size) X vocab_size
            #also force output to be in contiguous memory to make GPU computation easy.
            outputs=outputs.view(-1,5003).contiguous()
            #collapse sequence length and batch size dimensions and force input
            #to be (seq_length * batch_size)
            #also force input to be in contiguous memory to make GPU computation easy.          
            label_sentence = label_sentence.view(-1).contiguous()

            loss=criterion(outputs,label_sentence)

            # Backward pass:
            optimizer.zero_grad()
            # Loss is a variable, and calling backward on a Variable will
            # compute all the gradients that lead to that Variable taking on its
            # current value.
            loss.backward() 

            # Weight and bias updates.
            optimizer.step()
            cum_loss+=loss.data[0]
            counter=counter+paddedSeqs.size(1); #unlike the deep learning lab, 1 is the batch dimension here.
            total_words = total_words+paddedSeqs.size(0)*paddedSeqs.size(1) #num_words = words_per_sentence*num_sentences
            max_scores, max_labels = outputs.max(1)
            equal_tensor = torch.eq(max_labels,label_sentence)
            correct += torch.sum(equal_tensor.data)        
            # logging information.
            t.set_postfix(average_loss = cum_loss/counter, percent_correct = 100*correct/total_words)

        # Make a pass over the validation data.
        correct = 0.0
        total_words = 0.0
        cum_loss = 0.0
        counter = 0.0
        t = tqdm(valLoader, desc = 'Validation epoch %d' % epoch)
        network.eval()  # This is important to call before evaluating!
        for (i, (imgIds, paddedSeqs, seqLengths)) in enumerate(t):
            hidden_state = Variable(torch.Tensor(1, paddedSeqs.size(1), 512).zero_())
            input_sentence = torch.autograd.Variable(paddedSeqs[:-1])
            label_sentence = torch.autograd.Variable(paddedSeqs[1:])
            if use_gpu:
                label_sentence = label_sentence.cuda()
                input_sentence = input_sentence.cuda()
                hidden_state = hidden_state.cuda()

            # Forward pass:
            outputs, hidden_state  = model(input_sentence, hidden_state)
            #collapse sequence length and batch size dimensions and force output
            #to be (seq_length * batch_size) X vocab_size
            #also force output to be in contiguous memory to make GPU computation easy.
            outputs=outputs.contiguous().view(-1,5003)
            
            #collapse sequence length and batch size dimensions and force input
            #to be (seq_length * batch_size)
            #also force input to be in contiguous memory to make GPU computation easy.          
            label_sentence = label_sentence.contiguous().view(-1)

            loss=criterion(outputs,label_sentence)            
            cum_loss+=loss.data[0]
            counter=counter+paddedSeqs.size(1); #unlike the deep learning lab, 1 is the batch dimension here.
            total_words = total_words+paddedSeqs.size(0)*paddedSeqs.size(1) #num_words = words_per_sentence*num_sentences
            max_scores, max_labels = outputs.max(1)
            equal_tensor = torch.eq(max_labels,label_sentence)
            correct += torch.sum(equal_tensor.data)  
            # logging information.
            t.set_postfix(average_loss = cum_loss/counter, percent_correct = 100*correct/total_words)


# Now to the actual training call, notice how unlike previous experiments we are using here RMSprop which is a different type of optimizer that is often preferred for recurrent neural networks, although others such as SGD, and ADAM will also work. Additionally we are using nn.NLLLoss for the loss function, which is equivalent to the nn.CrossEntropyLoss function used before. The only difference is that nn.CrossEntropyLoss does the log_softmax operation for us, however in our implementation, we already applied log_softmax to the outputs of the model.

# In[ ]:


vocabularySize = len(trainData.vocabulary['word2id'])
#model = TextGeneratorModel(vocabularySize)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop(model.parameters(), lr = 0.001, momentum=0.001)

# Train the previously defined model.
train_rnn_model(model, criterion, optimizer, trainLoader, valLoader, n_epochs = 20, use_gpu = True)


# ## Lab Questions (10pts)
# <span><b>1.</b></span> (2pts) What is the number of parameters of the TextGeneratorModel? 

# In[8]:


#self.embedder = nn.Embedding(vocabularySize, 300)
#self.rnn = nn.RNN(300, 512, batch_first = False)
#self.classifier = nn.Linear(512, vocabularySize)


# The above three lines of code are the only "trainable" items in the TextGeneratorModel.  We have a vocabulary size of 5003 (5000 actual words + "START", "END", and "UNK"). The embedder is essentially just a lookup table that "hashes" an index in the 5003 long vocabulary to a 300-dimensional embedding vector. It therefore has $ 5003 * 300 = 1500900 $ parameters (no bias vector). The RNN itself does two things, maps input embeddings to hidden states as well as hidden states to hidden states. The embedding to hidden state mapping consists of a $ 300*512 = 153600 $ parameter linear transformation and a 512 parameter bias vector. The hidden state to hidden state mapping consists of a $ 512*512 = 262144 $ parameter linear transformation and a 512 parameter bias vector. The final classifier layer maps from hidden state to vocabulary size, so it consists of a
# $ 5003*512 = 2561536 $ parameter linear transformation and a 5003 parameter bias vector.
# 
# Adding all of these up:
# $ 1500900 + 153600 + 512 + 262144 + 512 + 2561536 + 5003 = 4484207$ total parameters!!!

# <span><b>2.</b></span> (4pts) Provide an implementation for the function train_rnn_model from section 3, this will be similar to the train_model function used in the previous lab. Then train the model and report a few sentences generated by your model. Use the following figure as reference to make sure you are using the right inputs and targets to train the model. The loss function between predictions and targets should be nn.CrossEntropyLoss(), so you might need to collapse the batch and sequence-length dimensions before passing them to the loss function.
# 
# <img src="rnn.png" width="80%"> 

# In[9]:


# implement train_rnn_model and then train the model using this function. 
# Show here a couple of sentences sampled from your model.
print(sample_sentence(model, use_cuda = True))
print(sample_sentence(model, use_cuda = True))
print(sample_sentence(model, use_cuda = True))
print(sample_sentence(model, use_cuda = True))
print(sample_sentence(model, use_cuda = True))


# <span><b>3. </b></span> (4pts) Create an ImageCaptioningModel class here that predicts a sentence given an input image. This should be an implementation of the model in this paper https://arxiv.org/pdf/1411.4555.pdf (See figure 3 in the paper). This model is very similar to the one implemented in this lab except that the first RNN cell gets the output of a CNN as its input. I'm also illustrating it below using a figure similar to the one in the previous question. For the CNN use Resnet-18. Note: You do not need to train this model, only define it. Feel free to start from the code for the TextGeneratorModel. <img src="im2text.png" width="80%">

# In[ ]:


import torchvision.models as models
resnet18 = models.resnet18(pretrained = True)
def resnetOutput(img):
    return resnet18(img)
class ImageCaptioningModel(nn.Module):
    # The model has FOUR layers: 
    #   
    #    1. An input translation later that takes outputs from Resnet-18
    #       and maps them to the vocabulary dimension
    #    2. An Embedding layer that turns a sequence of word ids into 
    #       a sequence of vectors of fixed size: embeddingSize.
    #    3. An RNN layer that turns the sequence of embedding vectors into 
    #       a sequence of hiddenStates.
    #    4. A classification layer that turns a sequence of hidden states into a 
    #       sequence of softmax outputs.
        
        
    def __init__(self, vocabularySize):
        super(ImageCaptioningModel, self).__init__()
        # See documentation for nn.Embedding here:Embedding
        self.image_embedder = nn.Linear(1000, 300)
        self.embedder = nn.Embedding(vocabularySize, 300)
        self.rnn = nn.RNN(300, 512, batch_first = False)
        self.classifier = nn.Linear(512, vocabularySize)
        self.vocabularySize = vocabularySize

    # The forward pass makes the sequences go through the three layers defined above.
    def forward(self, images, paddedSeqs, initialHiddenState):
        #get the output of resnet18
        z = resnetOutput(images)
        #extract the sequences as normal
        batchSequenceLength = paddedSeqs.size(0)  # 0-dim is sequence-length-dim.
        batchSize = paddedSeqs.size(1)  # 1-dim is batch dimension.
        
        #use a newly added linear layer as an "embedder" for the image.
        convnet_embedding = self.image_embedder(z)
        #embed the word IDs as normal
        seq_embedding = self.embedder(paddedSeqs)
        
        #append the embedding from the new linear layer to built-in 
        #embedding at the 0th word in each sentence in the batch. (seq length dimension is 0)
        embeddingVectors = torch.cat(convnet_embedding,seq_embedding[0])
        
        #carry on with the rest of the RNN as normal
        # Pass the sequence of word embeddings to the RNN.
        rnnOutput, finalHiddenState = self.rnn(embeddingVectors, initialHiddenState)
        
        # Collapse the batch and sequence-length dimensions in order to use nn.Linear.
        flatSeqOutput = rnnOutput.view(-1, 512)
        predictions = self.classifier(flatSeqOutput)
        
        # Expand back the batch and sequence-length dimensions and return. 
        return predictions.view(batchSequenceLength, batchSize, self.vocabularySize),                finalHiddenState


# ### Optional Questions (8pts)

# <span><b>1. </b></span> (1pts) What is the number of parameters of the ImageCaptioningModel from Q3?

# In[ ]:


# Show how did you come up with that number here.


# <span><b>2. </b></span> (3pts) Modify the TextGeneratorModel to use an LSTM instead, and retrain the model. Report results using this model.

# In[ ]:


print(sample_sentence(model, use_cuda = True))
print(sample_sentence(model, use_cuda = True))
print(sample_sentence(model, use_cuda = True))
print(sample_sentence(model, use_cuda = True))
print(sample_sentence(model, use_cuda = True))


# <span><b>3. </b></span> (4pts) In this question, you will have to reconstruct an input image from its activations. I will not provide you with the image, only the activation values obtained for a certain layer. You will have access to the code that was used to compute these activations. You will have to use back-propagation to reconstruct the input image. Show the reconstructed input image and tell us who is in the picture. Note: Look at the content reconstruction from outputs performed in https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Gatys_Image_Style_Transfer_CVPR_2016_paper.html

# In[12]:


import torchvision.models as models


# In[13]:


import pickle
import torchvision.transforms as transforms
from PIL import Image

preprocessFn = transforms.Compose([transforms.Scale(256), 
                                   transforms.CenterCrop(224), 
                                   transforms.ToTensor(), 
                                   transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                                                        std=[0.229, 0.224, 0.225])])

def model_F(input, model, kOutput = 19):
    prev_input = input
    for layer_id in range(0, kOutput + 1):
        current_input = model.features[layer_id](prev_input)
        prev_input = current_input
    return current_input
def file_to_image(filename):
    rtn = preprocessFn(Image.open(filename).convert('RGB'))
    rtn = (rtn.unsqueeze(0))
    return rtn


# In[15]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

def imshow(img):
    # convert torch tensor to PIL image and then show image inline.
    img = transforms.ToPILImage()(img[0].cpu() * 0.5 + 0.5) # denormalize tensor before convert
    plt.imshow(img, aspect = None)
    plt.axis('off')
    plt.gcf().set_size_inches(4, 4)
    plt.show()
class ImageModel(nn.Module):
    def __init__(self, image):
        super(ImageModel, self).__init__()
        self.image =  nn.Parameter(image,True)
    def forward(self, model):
        return model_F(self.image, model)
def train_image(img_model, cnn, criterion, optimizer, target, num_iterations = 10, use_gpu = False):
    img_model.train()
    cnn.eval()
    for i in range(0, num_iterations):
        cnn_out = img_model(cnn).cuda()
        loss=criterion(cnn_out,target)
        optimizer.zero_grad()
        loss.backward() 
        optimizer.step()
        img_model.image.data.clamp_(0, 1)
        print('current loss:', loss)
    print('final loss:', loss)
    return img_model

target = torch.autograd.Variable(torch.load(open('layer-19-output.p'))).cuda()
print(target.data.size())
    
vgg = models.vgg16(pretrained = True).cuda()
#start with Bill Clinton's White House portrait as an initial guess
guess_image = file_to_image('bill.jpg')
#guess_image = (torch.randn([1, 3, 224, 224])+torch.randn([1, 3, 224, 224])).clamp_(0, 1)
old_image = guess_image.clone()
image_criterion = nn.MSELoss().cuda()
new_image_model  = ImageModel(guess_image).cuda()
image_optimizer = torch.optim.SGD(new_image_model.parameters(), lr=0.3, momentum=0.9)
train_image(new_image_model, vgg, image_criterion, image_optimizer, target, num_iterations = 20000,use_gpu = False)

# Your solution goes here. Show the reconstructed input and tell us who is depicted in the incognito.jpg image.


# In[16]:


imshow(new_image_model.image.cpu().data)


# It's Grace Hopper!!! 
# https://www.biography.com/.image/t_share/MTE5NTU2MzE2NjYxNTE1Nzg3/grace-hopper-21406809-1-402.jpg
# 
# 
# 

# 
# 

# <div style="font-size:0.8em;color:#888;text-align:center;padding-top:20px;">If you find any errors or omissions in this material please contact me at vicente@virginia.edu</div>
