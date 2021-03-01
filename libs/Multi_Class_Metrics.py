import tensorflow as tf

class _ConfusionMatrixMetric(tf.keras.metrics.Metric):
    """
    A custom Keras metric to compute the running average of the confusion matrix
    """
    def __init__(self, num_classes, **kwargs):
        super(_ConfusionMatrixMetric,self).__init__(**kwargs) # handles base args (e.g., dtype)
        self.num_classes=num_classes
        self.total_cm = self.add_weight("total", shape=(num_classes,num_classes), initializer="zeros")
        
    def reset_states(self):
        for s in self.variables:
            s.assign(tf.zeros(shape=s.shape))
            
    def update_state(self, y_true, y_pred,sample_weight=None):
        self.total_cm.assign_add(self.confusion_matrix(y_true,y_pred))
        return self.total_cm
        
    def result(self):
        return self.calculate_metric()
    
    def confusion_matrix(self,y_true, y_pred):
        """
        Make a confusion matrix
        """
        y_pred=tf.argmax(y_pred,1)
        y_true=tf.argmax(y_true,1)
        
        # columns represent the prediction labels and the rows represent the real label
        cm=tf.math.confusion_matrix(y_true,y_pred,dtype=tf.float32,num_classes=self.num_classes)
        return cm
    
    def calculate_metric(self):
        return None
    
    def fill_output(self,output):
        results=self.result()
        output['None']=results
        
    def get_config(self):
        config = {
            'num_classes': self.num_classes
            }
        base_config = super(_ConfusionMatrixMetric, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
        
            
            
class MultiClassSpecificity(_ConfusionMatrixMetric):
    """
    A custom Keras metric to compute the running average of the confusion matrix
    """
    def __init__(self, num_classes, pos_ind,average='macro', **kwargs):
        """
        param average: 
            'macro': averages metrics of classes in pos_ind
            'micro': NotImplementedError
        """
        if not 'name' in kwargs.keys():
            name='multi_class_specificity'
            kwargs['name']=name
        
        super(MultiClassSpecificity,self).__init__( num_classes=num_classes,**kwargs) # handles base args (e.g., dtype)
        self.pos_ind = pos_ind
        self.average = average
        if average=='micro':
            raise NotImplementedError
        self.total_cm = self.add_weight("total", shape=(num_classes,num_classes), initializer="zeros")
        
    
    def calculate_metric(self):
        "returns precision, recall and f1 along with overall accuracy"
        cm=self.total_cm
        diag_part=tf.linalg.diag_part(cm)

        N  = -tf.reduce_sum(cm,0)+tf.reduce_sum(cm)
        TN = -tf.reduce_sum(cm,0)-tf.reduce_sum(cm,1)+diag_part+tf.reduce_sum(cm)
        specificity = TN/N
        if self.average=='macro':
            pos_class_specificity = 0
            for ind in self.pos_ind:
                pos_class_specificity+=specificity[ind]
            pos_class_specificity/=len(self.pos_ind)
            return pos_class_specificity
        
    def fill_output(self,output):
        results=self.result()
        output['{}_specificity'.format(self.average)]=results
        
    def get_config(self):
        config = {
            'pos_ind': list(self.pos_ind),
            'average': self.average
            }
        base_config = super(MultiClassSpecificity, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MultiClassRecall(_ConfusionMatrixMetric):
    """
    A custom Keras metric to compute the running average of the confusion matrix
    """
    def __init__(self, num_classes,pos_ind,average='macro', **kwargs):
        if not 'name' in kwargs.keys():
            name='multi_class_recall'
            kwargs['name']=name
        # add arguments to kwarg because they are needed for the config
        super(MultiClassRecall,self).__init__(num_classes=num_classes,**kwargs) # handles base args (e.g., dtype)
        self.num_classes=num_classes
        self.pos_ind = pos_ind
        self.average = average
        self.total_cm = self.add_weight("total", shape=(num_classes,num_classes), initializer="zeros")
        
    
    def calculate_metric(self):
        "returns precision, recall and f1 along with overall accuracy"
        cm=self.total_cm
        diag_part=tf.linalg.diag_part(cm)
        #precision=diag_part/(tf.reduce_sum(cm,0)+tf.constant(1e-15))
        # tf.reduce_sum(cm,1)[i] predicted samples for i
        recall=diag_part/(tf.reduce_sum(cm,1)+tf.constant(1e-15))
        #f1=2*precision*recall/(precision+recall+tf.constant(1e-15))
        if self.average=='macro':
            pos_class_recall = 0
            for ind in self.pos_ind:
                pos_class_recall+=recall[ind]
            pos_class_recall/=len(self.pos_ind)
            return pos_class_recall
        
    def fill_output(self,output):
        results=self.result()
        output['{}_recall'.format(self.average)]=results

    def get_config(self):
        config = {
            'pos_ind': list(self.pos_ind),
            'average': self.average
            }
        base_config = super(MultiClassRecall, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

            
class MultiClassF1(_ConfusionMatrixMetric):
    """
    A custom Keras metric to compute the running average of the confusion matrix
    """
    def __init__(self, num_classes,pos_ind,average='macro', **kwargs):
        if not 'name' in kwargs.keys():
            name='multi_class_F1'
            kwargs['name']=name
            
        super(MultiClassF1,self).__init__(num_classes=num_classes,**kwargs) # handles base args (e.g., dtype)
        self.num_classes=num_classes
        self.pos_ind = pos_ind
        self.average = average
        self.total_cm = self.add_weight("total", shape=(num_classes,num_classes), initializer="zeros")
         
    def calculate_metric(self):
        "returns precision, recall and f1 along with overall accuracy"
        cm=self.total_cm
        diag_part=tf.linalg.diag_part(cm)
        precision=diag_part/(tf.reduce_sum(cm,0)+tf.constant(1e-15))
        # tf.reduce_sum(cm,1)[i] predicted samples for i
        recall=diag_part/(tf.reduce_sum(cm,1)+tf.constant(1e-15))
        f1=2*precision*recall/(precision+recall+tf.constant(1e-15))
        if self.average=='macro':
            pos_class_f1 = 0
            for ind in self.pos_ind:
                pos_class_f1+=f1[ind]
            pos_class_f1/=len(self.pos_ind)
            return pos_class_f1
        
    def fill_output(self,output):
        results=self.result()
        output['{}_F1'.format(self.average)]=results
            
    def get_config(self):
        config = {
            'pos_ind': list(self.pos_ind),
            'average': self.average
            }
        base_config = super(MultiClassF1, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


