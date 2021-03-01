from sklearn import metrics
import numpy as np
print('calc scores updata')

def get_confusion_matrix(y_true,y_pred):
    from sklearn.metrics import confusion_matrix
    conf_matrix = confusion_matrix(y_true.flatten(),y_pred.flatten())
    return conf_matrix

def _class_selection(classes):
    """
    creates index pairs for confusion matrix selection
    """
    index = []
    for c1 in classes:
        for c2 in classes:
            index+=[[c1,c2]]
    index = np.array(index)
    return index
    
def accuracy_from_conf_matrix(conf_matrix,classes):
    index = _class_selection(classes)
    selected_classes = conf_matrix[index[:,0],index[:,1]]
    hits_index = index[index[:,0]==index[:,1]]
    hits = conf_matrix[hits_index[:,0],hits_index[:,1]]
    return np.sum(hits)/np.sum(selected_classes)