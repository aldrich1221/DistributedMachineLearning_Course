
#######
import keras
import numpy as np
import xml.etree.ElementTree as ET
import os
import tensorflow as tf


####

Gesture=['Press_Double_Tap','Press_Tap','Scale','Translation','Rotation']
#Gesture=['Press_Tap','Translation']
NumOfGesture=len(Gesture)
batch_size=128
nb_classes=NumOfGesture
nb_epoch=50
 
img_rows,img_cols=10,10
FingerNum=10
TouchPointNum=15
nb_filters=32
pool_size=(2,2)
kernel_size=(3,3)
channel=3

def one_hot_encode(x,NumOfGesture):
    """
        argument
            - x: a list of labels
        return
            - one hot encoding matrix (number of labels, number of class)
    """
    encoded = np.zeros((len(x), NumOfGesture))
    
    for idx, val in enumerate(x):
        encoded[idx][val] = 1
    
    return encoded
########data pre-process#########

AllDataX_positionX=list()
AllDataX_positionY=list()
AllDataX_Timestamp=list()
#AllDataY=[[],[],[],[],[]]
AllDataY=list()

AllDataPredictX_positionX=list()
AllDataPredictX_positionY=list()
AllDataPredictX_Timestamp=list()
#AllDataPredictY=[[],[],[],[],[]]
AllDataPredictY=list()

for igesture in range(NumOfGesture):
    gesture=Gesture[igesture]

    path ="./TrainData/"+str(gesture)+"/"
    UserCount = -1
    for fn in os.listdir(path): #fn 表示的是文件名
        UserCount = UserCount+1
    for iUser in range(1,UserCount):
        if iUser<20: #training data
            TrialCount=0;
            filepath='./TrainData/'+str(gesture)+'/'+str(iUser)
            for fn in os.listdir(filepath): #fn 表示的是文件名
                TrialCount = TrialCount+1
            for iTrial in range(1,TrialCount):
                
                tree = ET.parse('./TrainData/'+str(gesture)+'/'+str(iUser)+'/'+str(gesture)+'-'+str(iTrial)+'.xml')
                root = tree.getroot()
                datacount=len(root[1].findall('data'))
                TouchID=[];
                TimeStampArray=[];
                for i in range(datacount):
                    TouchID.append(int(root[1][i].attrib['id']))
                    TimeStampArray.append(int(root[1][i].attrib['t']))
                TouchID=list(set(TouchID))
                TimeStampArray=np.sort(list(set(TimeStampArray)))  #unique所有時間戳記
                
                
                TempArrayX=np.zeros((FingerNum,len(TimeStampArray)))
                TempArrayY=np.zeros((FingerNum,len(TimeStampArray)))
                TempArrayTimeStamp=np.zeros((FingerNum,len(TimeStampArray)))
                TrailDataX=[[],[],[],[],[],[],[],[],[],[]]
                TrailDataY=[[],[],[],[],[],[],[],[],[],[]]
                TrailDataTimeStamp=[[],[],[],[],[],[],[],[],[],[]]
                for t_timestamp in range(len(TimeStampArray)):
                    for iFingerId in range(len(TouchID)):
                        fingerid=TouchID[iFingerId]
                        TempArrayX[iFingerId][t_timestamp]=0.0
                        TempArrayY[iFingerId][t_timestamp]=0.0
                        TempArrayTimeStamp[iFingerId][t_timestamp]=0.0
                        for elem in root[1].iter(tag='data'):
                            #print(elem.tag, elem.attrib['t'])
                            if(fingerid==int(elem.attrib['id'])):
                                if(int(elem.attrib['t'])==int(TimeStampArray[t_timestamp])):
                                    TempArrayX[iFingerId][t_timestamp]=elem.attrib['xNorm']
                                   
                                    TempArrayY[iFingerId][t_timestamp]=elem.attrib['yNorm']
                                    TempArrayTimeStamp[iFingerId][t_timestamp]=TimeStampArray[t_timestamp]

                ##Scale TimeStampArray
#                for t_timestamp in range(len(TimeStampArray)):
#                    for iFingerId in range(len(TouchID)):
#                        if TempArrayTimeStamp[iFingerId][t_timestamp]!=0.0:
#                            TempArrayTimeStamp[iFingerId][t_timestamp]=(TempArrayTimeStamp[iFingerId][t_timestamp]-TimeStampArray[0])/(TimeStampArray[-1]-TimeStampArray[0])
                ##
                for iFingerId in range(FingerNum):
                     if(iFingerId<len(TouchID)):
                         if(len(TempArrayX[iFingerId]))>=TouchPointNum:
                             for i in range(TouchPointNum):
                                 index=i*int(len(TempArrayX[iFingerId])/TouchPointNum)
                                 
                            
                                 TrailDataX[iFingerId].append(TempArrayX[iFingerId][index])
                                 TrailDataY[iFingerId].append(TempArrayY[iFingerId][index])
                                 TrailDataTimeStamp[iFingerId].append(TempArrayTimeStamp[iFingerId][index])
                                 
                         else:
                             for i in range(TouchPointNum):
                                 if(i<len(TempArrayX[iFingerId])):
                                     TrailDataX[iFingerId].append(TempArrayX[iFingerId][i])
                                     TrailDataY[iFingerId].append(TempArrayY[iFingerId][i])
                                     TrailDataTimeStamp[iFingerId].append(TempArrayTimeStamp[iFingerId][i])
                                 else:
                                     TrailDataX[iFingerId].append(0)
                                     TrailDataY[iFingerId].append(0)
                                     TrailDataTimeStamp[iFingerId].append(0)
                     else:
                         for i in range(TouchPointNum):
                             TrailDataX[iFingerId].append(0)
                             TrailDataY[iFingerId].append(0)
                             TrailDataTimeStamp[iFingerId].append(0)
            
                AllDataX_positionX.append(TrailDataX)
                AllDataX_positionY.append(TrailDataY)
                AllDataX_Timestamp.append(TrailDataTimeStamp)
                AllDataY.append(igesture)
                #AllDataY[igesture]=igesture
        else:
            TrialCount=0;
            filepath='./Itekube/'+str(gesture)+'/'+str(iUser)
            for fn in os.listdir(filepath): #fn 表示的是文件名
                TrialCount = TrialCount+1
            for iTrial in range(1,TrialCount):
                
                tree = ET.parse('./Itekube/'+str(gesture)+'/'+str(iUser)+'/'+str(gesture)+'-'+str(iTrial)+'.xml')
                root = tree.getroot()
                datacount=len(root[1].findall('data'))
                TouchID=[];
                TimeStampArray=[];
                for i in range(datacount):
                    TouchID.append(int(root[1][i].attrib['id']))
                    TimeStampArray.append(int(root[1][i].attrib['t']))
                TouchID=list(set(TouchID))
                TimeStampArray=np.sort(list(set(TimeStampArray)))  #unique所有時間戳記
                
                
                TempArrayX=np.zeros((FingerNum,len(TimeStampArray)))
                TempArrayY=np.zeros((FingerNum,len(TimeStampArray)))
                TempArrayTimeStamp=np.zeros((FingerNum,len(TimeStampArray)))
                TrailDataX=[[],[],[],[],[],[],[],[],[],[]]
                TrailDataY=[[],[],[],[],[],[],[],[],[],[]]
                TrailDataTimeStamp=[[],[],[],[],[],[],[],[],[],[]]
                for t_timestamp in range(len(TimeStampArray)):
                    for iFingerId in range(len(TouchID)):
                        fingerid=TouchID[iFingerId]
                        TempArrayX[iFingerId][t_timestamp]=0.0
                        TempArrayY[iFingerId][t_timestamp]=0.0
                        TempArrayTimeStamp[iFingerId][t_timestamp]=0.0
                        for elem in root[1].iter(tag='data'):
                            #print(elem.tag, elem.attrib['t'])
                            if(fingerid==int(elem.attrib['id'])):
                                if(int(elem.attrib['t'])==int(TimeStampArray[t_timestamp])):
                                    TempArrayX[iFingerId][t_timestamp]=elem.attrib['xNorm']
                                   
                                    TempArrayY[iFingerId][t_timestamp]=elem.attrib['yNorm']
                                    TempArrayTimeStamp[iFingerId][t_timestamp]=TimeStampArray[t_timestamp]

                ##Scale TimeStampArray
#                for t_timestamp in range(len(TimeStampArray)):
#                    for iFingerId in range(len(TouchID)):
#                        if TempArrayTimeStamp[iFingerId][t_timestamp]!=0.0:
#                            TempArrayTimeStamp[iFingerId][t_timestamp]=(TempArrayTimeStamp[iFingerId][t_timestamp]-TimeStampArray[0])/(TimeStampArray[-1]-TimeStampArray[0])
#                ##
                for iFingerId in range(FingerNum):
                     if(iFingerId<len(TouchID)):
                         if(len(TempArrayX[iFingerId]))>=TouchPointNum:
                             for i in range(TouchPointNum):
                                 index=i*int(len(TempArrayX[iFingerId])/TouchPointNum)
                                 
                            
                                 TrailDataX[iFingerId].append(TempArrayX[iFingerId][index])
                                 TrailDataY[iFingerId].append(TempArrayY[iFingerId][index])
                                 TrailDataTimeStamp[iFingerId].append(TempArrayTimeStamp[iFingerId][index])
                                 
                         else:
                             for i in range(TouchPointNum):
                                 if(i<len(TempArrayX[iFingerId])):
                                     TrailDataX[iFingerId].append(TempArrayX[iFingerId][i])
                                     TrailDataY[iFingerId].append(TempArrayY[iFingerId][i])
                                     TrailDataTimeStamp[iFingerId].append(TempArrayTimeStamp[iFingerId][i])
                                 else:
                                     TrailDataX[iFingerId].append(0)
                                     TrailDataY[iFingerId].append(0)
                                     TrailDataTimeStamp[iFingerId].append(0)
                     else:
                         for i in range(TouchPointNum):
                             TrailDataX[iFingerId].append(0)
                             TrailDataY[iFingerId].append(0)
                             TrailDataTimeStamp[iFingerId].append(0)
            
                AllDataPredictX_positionX.append(TrailDataX)
                AllDataPredictX_positionY.append(TrailDataY)
                AllDataPredictX_Timestamp.append(TrailDataTimeStamp)
                AllDataPredictY.append(igesture)
                    
                

########data pre-process#########
#AllDataPredictY=np.array(AllDataPredictY)
#AllDataY=np.array(AllDataY)
##train
AllDataY=one_hot_encode(AllDataY,NumOfGesture)
AllDataX_positionX=np.array(AllDataX_positionX)
AllDataX_positionY=np.array(AllDataX_positionY)
AllDataX_Timestamp=np.array(AllDataX_Timestamp)
AllDataX=np.array([AllDataX_positionX,AllDataX_positionY,AllDataX_Timestamp])
A=AllDataX.transpose(1,2,3,0)
####Test

AllDataPredictX_positionX=np.array(AllDataPredictX_positionX)
AllDataPredictX_positionY=np.array(AllDataPredictX_positionY)
AllDataPredictX_Timestamp=np.array(AllDataPredictX_Timestamp)
AllDataXPredict=np.array([AllDataPredictX_positionX,AllDataPredictX_positionY,AllDataPredictX_Timestamp])
AllDataPredictY=one_hot_encode(AllDataPredictY,NumOfGesture)

B=AllDataXPredict.transpose(1,2,3,0)


# =============================================================================
# ########Model Setting###########
# #CONV128 
# #POOL 1X2 only time dimension
# #CONV128,POOL 1X2 only time dimension
# #CONV256,Fully connected
def InitWeights(shape):
    return tf.Variable(tf.random_normal(shape,stddev=0.01))
     

X=tf.placeholder("float",[None,img_rows,TouchPointNum,channel],name='myImage')
Y=tf.placeholder("float",[None,NumOfGesture],name='myLabel')
 
w=InitWeights([3,3,channel,128])
w2=InitWeights([3,3,128,128])
w3=InitWeights([3,3,128,256])
w4=InitWeights([256*4,256*NumOfGesture])
w_o=InitWeights([256*NumOfGesture,NumOfGesture])
 
def model(X,w,w2,w3,w4,w_o):
    print(X)     
    layer1a=tf.nn.relu((tf.nn.conv2d(X,w,strides=[1,1,1,1],padding='SAME')))
    print("Layer1a shape: ")
    print(layer1a)
    layer1=tf.nn.max_pool(layer1a,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    #layer1=tf.nn.dropout(layer1,p_keep_conv)
    layer1=tf.nn.dropout(layer1,0.1)
    
    print("Layer1 shape: ")
    print(layer1.shape)
    ###
    layer2a=tf.nn.relu((tf.nn.conv2d(layer1,w2,strides=[1,1,1,1],padding='SAME')))
    print("Layer2a shape: ")
    print(layer2a.shape)
    layer2=tf.nn.max_pool(layer2a,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    #layer2=tf.nn.dropout(layer2,p_keep_conv)
    layer2=tf.nn.dropout(layer2,0.1)
    print("Layer2 shape: ")
    print(layer2.shape)
    ###
    layer3a=tf.nn.relu((tf.nn.conv2d(layer2,w3,strides=[1,1,1,1],padding='SAME')))
    print("Layer3a shape: ")
    print(layer3a.shape)
    layer3=tf.nn.max_pool(layer3a,ksize=[1,2,2,1],strides=[1,1,2,1],padding='VALID')
    print("Layer3 shape: ")
    print(layer3.shape)
    layer3=tf.reshape(layer3,[-1,w4.get_shape().as_list()[0]])
    #layer3=tf.nn.dropout(layer3,p_keep_conv)
    layer3=tf.nn.dropout(layer3,0.1)
    
     
    print("Layer3 shape: ")
    print(layer3.shape)
    ###
    layer4=tf.nn.relu(tf.matmul(layer3,w4))
    #layer4=tf.nn.dropout(layer4,p_keep_hidden)
    layer4=tf.nn.dropout(layer4,0.5)
     
    print("Layer4 shape: ")
    print(layer4.shape)
     
     ###
    pyx=tf.matmul(layer4,w_o)
    print("Output shape: ")
    print(pyx.shape)
    return pyx
 
 #p_keep_conv=tf.placeholder("float",name='myP_Keep_Conv')
 #p_keep_hidden=tf.placeholder("float",name='myP_Keep_Hidden')
py_x=model(X,w,w2,w3,w4,w_o)
             
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=py_x,labels=Y))
train_op=tf.train.RMSPropOptimizer(0.001,0.9).minimize(cost)
predict_op=tf.argmax(py_x,1,name='predict')
# ########Model Setting###########
# =============================================================================


########Model2 Setting###########
#batch_size=128
#nb_classes=5
#nb_epoch=12
## 
#img_rows,img_cols=10,10
#nb_filters=32
#pool_size=(2,2)
#kernel_size=(3,3)
#def InitWeights(shape):
#    return tf.Variable(tf.random_normal(shape,stddev=0.01))
#    
#
#X=tf.placeholder("float",[None,10,10,2],name='myImage')
#Y=tf.placeholder("float",[None,5],name='myLabel')
#w=InitWeights([3,3,2,128])
#w2=InitWeights([128*2*2,625])
#w_o=InitWeights([625,5])
#
#def model2(X,w,w2,w_o,p_keep_conv,p_keep_hidden):
#    print(X)     
#    layer1a=tf.nn.relu((tf.nn.conv2d(X,w,strides=[1,1,1,1],padding='SAME')))
#    print("Layer1a shape: ")
#    print(layer1a)
#    layer1=tf.nn.max_pool(layer1a,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
#    layer1=tf.reshape(layer1,[-1,w2.get_shape().as_list()[0]])
#    layer1=tf.nn.dropout(layer1,p_keep_conv)
#    print("Layer1 shape: ")
#    print(layer1.shape)
#     ###
#     ###
#    layer2=tf.nn.relu(tf.matmul(layer1,w2))
#    layer2=tf.nn.dropout(layer2,p_keep_hidden)
#    
#     ###
#     
#    pyx=tf.matmul(layer2,w_o)
#    print("Output shape: ")
#    print(pyx.shape)
#    return pyx
#
#p_keep_conv=tf.placeholder("float",name='myP_Keep_Conv')
#p_keep_hidden=tf.placeholder("float",name='myP_Keep_Hidden')
#py_x=model2(X,w,w2,w_o,p_keep_conv,p_keep_hidden)
##             
#cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x,labels=Y))
#train_op=tf.train.RMSPropOptimizer(0.001,0.9).minimize(cost)
#predict_op=tf.argmax(py_x,1,name='predict')
########Model2 Setting###########



########Model Training###########
#from tensorflow import freeze_graph 
from tensorflow.python.tools.freeze_graph import freeze_graph 
from tensorflow.python.framework.graph_util import convert_variables_to_constants

with tf.Session() as sess:
   
    tf.global_variables_initializer().run()
    for i in range(nb_epoch):
        for start ,end in zip(range(0,len(A),batch_size),range(batch_size,len(A)+1,batch_size)):
            sess.run(train_op,feed_dict={X:A[start:end],Y:AllDataY[start:end].reshape(batch_size,5)})
        print(i,np.mean(np.argmax(AllDataPredictY,axis=1)==sess.run(predict_op,feed_dict={X:B})))
        saver=tf.train.Saver()
        # save graph definition somewhere
        tf.train.write_graph(sess.graph, '', './OutputModel/Fifth.pbtxt')
        # save the weights
        saver.save(sess, './OutputModel/Fifth.ckpt')
    
    output_graph_def = convert_variables_to_constants(sess, sess.graph_def, output_node_names=['predict'])
    with tf.gfile.FastGFile('./OutputModel/FifthModel.pb', mode='wb') as f:
        f.write(output_graph_def.SerializeToString())
     
#
print("Now pb to mlmodel")

import tfcoreml
#
#tfcoreml.convert(tf_model_path='./OutputModel/ForthModel.pb',
#                    mlmodel_path='./OutputModel/ForthModel.mlmodel',
#                    input_name_shape_dict={"myImage":[3,10,10]},
#                    output_feature_names = ['predict:0'])
 #,input_name_shape_dict={'myImage':[3,10,10]})


tfcoreml.convert(tf_model_path='./OutputModel/FifthModel.pb',
                    mlmodel_path='./OutputModel/FifthModel.mlmodel',
                    output_feature_names = ['predict:0'])


#tfcoreml.convert(tf_model_path='./OutputModel/ForthModel.pb',
#                    mlmodel_path='./OutputModel/ForthModel.mlmodel')