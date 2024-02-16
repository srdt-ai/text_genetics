import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as pim
import time
from skimage.feature import greycomatrix, greycoprops, hog, local_binary_pattern
import scipy
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.cluster import KMeans

class QueryMethod1():

    def __init__(self,n,base_images,query_descriptor,base_descriptor,Distance):
        self.n = n
        self.base_images = base_images
        self.query_descriptor = query_descriptor
        self.base_descriptor = base_descriptor
        self.Distance = Distance


    def retrieve_images(self,query_name):

        vector1 = self.query_descriptor[query_name]
        distances = []
        for im2,name in self.base_images:
            vector2 = self.base_descriptor[name]
            if self.Distance == 'euclidian':
                d = np.linalg.norm(vector1-vector2)
            if self.Distance == 'cosine':
                d = 1 - vector1.T@vector2/(np.linalg.norm(vector1)*np.linalg.norm(vector2))
            if self.Distance == 'spatial':
                d = np.mean(scipy.spatial.distance.cdist(vector1,vector2))
            if self.Distance == 'chi-squared':
                d = 0.5*sum([(vector1[m]-vector2[m])**2/(vector1[m]+vector2[m]+1) for m in range(len(vector1))])
            distances.append(d)

        # attention: plusieurs documents ont des vecteurs descripteurs qui retournent la même distance 
        # donc la même image est retournée plusieurs fois

        top_distances = np.sort(distances)[:self.n]
        top_images = []
        for d in np.unique(top_distances) :
            indexes = np.where(distances==d)[0]
            for idx in indexes:
                if len(top_images)<self.n:
                    top_images.append(self.base_images[idx])
            
        return top_images


    def precision(self,retrieved, relevent):
        TP = 0  # docs pertinents
        FP = 0  # docs non pertinents
        relevent_names = [name for (img,name) in relevent]
        for img,name in retrieved :
            if name in relevent_names:
                TP+=1
            else :
                FP+=1
        return TP/(TP+FP)


    def rappel(self,retrieved, relevent):
        TP = 0  # docs pertinents
        FN = 0  # docs non pertinents
        retrieved_names = [name for (img,name) in retrieved]
        for img,name in relevent:
            if name in retrieved_names:
                TP+=1
            else :
                FN+=1
        return TP/(TP+FN)


    def precision_recall_curv(self,retrieved,relevent):

        x = []
        y = []
        for k in range(1,len(retrieved)+1):
            y.append(self.precision(retrieved[:k],relevent))
            x.append(self.rappel(retrieved[:k],relevent))
        plt.scatter(x,y)
        plt.plot(x,y)
        return x,y


    def compute_map(self,query_images,relevent_images):
        mAP=0
        for query, name in query_images :
            relevent_names = [nam for (img,nam) in relevent_images[name]]
            retrieved = self.retrieve_images(name)
            nRelevent = len(relevent_names)
            AP = 0
            TP=0
            i=1
            for ret_im,ret_name in retrieved :
                if ret_name in relevent_names :
                    TP+=1
                    AP+= TP/i
                    i+=1
        mAP+=AP/nRelevent
        return mAP/len(query_images)
    
from os import listdir
from os.path import isfile, join

# liste contenant les noms des documents du dossier
all_images = [im for im in listdir('nom_repertoire') if isfile(join('nom_repertoire', im))]


query_images = []   # liste des couples (image,nom) requête 
base_images = []    # liste des couples (image,nom) de la base 
for i,im in enumerate(all_images) :
    img = pim.open('nom_repertoire/'+im)
    if im[-6:-4]=='00':
        query_images.append((img,im))
    else :
        base_images.append((img,im))


# fonction qui trouve les documenrs pertinents pour chaque doc en entrée
relevent_images = {}
for query,name in query_images :
    relevent_images[name] = []
    prefix = name[:4]
    for img,im in base_images:
        if im[:4]==prefix and im[5]!='0':
            relevent_images[name].append((img,im))

"""
# converting to gray scale
gray_query = []
for img, name in query_images :
    im = pim.open('nom_repertoire/'+name).convert('LA')
    im = np.asarray(im,dtype = np.int64)
    im = im[:,:,0]
    gray_query.append((im,name))

gray_base = []
for img, name in base_images :
    im = pim.open('nom_repertoire/'+name).convert('LA')
    im = np.asarray(im,dtype = np.int64)
    im = im[:,:,0]
    gray_base.append((im,name))"""

def flat(image):
    img = image.resize((16,16))
    img = np.asarray(img,dtype = np.int64)
    img = img.flatten()
    return img


def color_hist(image):

    img = np.asarray(image,dtype = np.int64)
    Rhist = np.histogram(img[:,:,0], bins=256, range=(0, 255))[0]
    Ghist = np.histogram(img[:,:,1], bins=256, range=(0, 255))[0]
    Bhist = np.histogram(img[:,:,2], bins=256, range=(0, 255))[0]
    colorHist = np.hstack((Rhist,Ghist,Bhist))
    return colorHist


def gray_hist(gray_image):
    img = np.asarray(gray_image,dtype = np.int64)
    return np.histogram(img, bins=256, range=(0, 255))[0]

query_images

im1 , name1 = query_images[0]
im1

Q = QueryMethod1(10,base_images)

i0 = 17
pertinence = Q.find_relevent_images(query_images[i0][1])
retour = Q.retrieve_images(query_images[i0][0])
Q.precision_recall_curv(retour,pertinence)

pertinence[0][1],pertinence[1][1]

plt.figure(figsize=(20,15))
n0 = len(pertinence)
plt.subplot(1,n0+1,1)
query = plt.imread('nom_repertoire/'+query_images[i0][1])
plt.title('query image')
plt.imshow(query)
plt.axis('off')

for j,(im2,name) in enumerate(pertinence) :
    plt.subplot(1,n0+1,j+2)
    answer = plt.imread('nom_repertoire/'+name)
    plt.title('rank {} relevent image'.format(j+1))
    plt.imshow(answer)
    plt.axis('off')

plt.figure(figsize=(15,5))
for j,(im2,name) in enumerate(retour) :
    plt.subplot(2,5,j+1)
    answer = plt.imread('nom_repertoire/'+name)
    plt.title('rank {} answer'.format(j+1))
    plt.imshow(answer)
    plt.axis('off')

    file = open('myresults.txt','w')

for query, name in query_images:
    print(name)
    file.write('{} 0 '.format(name))
    retrieved = Q.retrieve_images(query)
    
    for j,(img,nom) in enumerate(retrieved):
        file.write('{} {} '.format(nom,j+1))
    file.write('\n')  

def compute_map(query_images,Q):
    mAP=0
    for query, name in query_images :
        relevent = Q.find_relevent_images(name)
        relevent_names = [nam for (img,nam) in relevent]
        retrieved = Q.retrieve_images(query)
        nRelevent = len(relevent)
        AP = 0
        TP=0
        i=1
        for ret_im,ret_name in retrieved :
            if ret_name in relevent_names :
                TP+=1
                AP+= TP/i
            i+=1
    mAP+=AP/nRelevent
    return mAP/len(query_images)

for k in [1,5,10]:
    Q = QueryMethod1(k,base_images)
    print(compute_map(query_images,Q))

mAP=0
with open('myresults.txt','r') as results :
    for line in results:
        query = line[:10]
        relevent = Q.find_relevent_images(query)
        relevent_names = [name for (img,name) in relevent]
        nRelevent = len(relevent)
        AP = 0
        TP=0
        i=1
        while 13*i+10 < len(line) :
            retrieved = line[13*i:13*i+10]
            if retrieved in relevent_names :
                TP+=1
                AP+= TP/i
            i+=1
        mAP+=AP/nRelevent
mAP/len(query_images)

mAP=0
with open('myresults.txt','r') as results :
    for line in results:
        query = line[:10]
        relevent = Q.find_relevent_images(query)
        relevent_names = [name for (img,name) in relevent]
        nRelevent = len(relevent)
        AP = 0
        TP=0
        i=1
        while 13*i+10 < len(line) :
            retrieved = line[13*i:13*i+10]
            if retrieved in relevent_names :
                TP+=1
                AP+= TP/i
            i+=1
        mAP+=AP/nRelevent
mAP/len(query_images)

nom_repertoire_map('myresults.txt')



    