
## Keyhan Azarjoo
## K.azarjoo@tees.ac.uk
## B1674080@tees.ac.uk
## Keyhanazarjoo@gmail.com

## Teesside University
## Final Project
## Superviser : Annalisa Occhipinti

####################                                                                        #################### 
####################                                                                        #################### 
####################          3D and 2D object Detection in Lidar Point Cloud data          #################### 
####################                                                                        #################### 
####################                                                                        #################### 






#%%
##########          Cleaning and Clustering          ##########
##########          Cleaning and Clustering          ##########
##########          Cleaning and Clustering          ##########
##########          Cleaning and Clustering          ##########
##########          Cleaning and Clustering          ##########

#%%

import open3d as o3d
import numpy as np
import matplotlib as plt

print("Loading Point Cloud")
pcd = o3d.io.read_point_cloud(r"C:\Users\jawad_pl\OneDrive - Teesside University\Project Lidar\Learning\Object detection\segmentation\Uni Samples\IDTC point cloud.ply")
#pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=16), fast_normal_computation=True)
pcdO = pcd
print("Point Cloud has been loaded\n")

#o3d.visualization.draw_geometries([pcd])


print("==========================================================================")

#Data Cleaning
print("Start Cleaning Point Cloud\n")

#voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.02)
#uni_down_pcd = pcd.uniform_down_sample(every_k_points=1)
#For comparison, uniform_down_sample can downsample point cloud by collecting every n-th points.

def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])


pcdcls, ind = pcd.remove_statistical_outlier(nb_neighbors=20,
                                                   std_ratio=2)
print("\nStatistical oulier removal\n")
#display_inlier_outlier(pcd, ind)
#TT = pcd.select_by_index(ind, invert=True)
#TT2 = pcd.select_by_index(ind)
print('Original Points Count : ' + str(len(np.asarray(pcd.points))))
print('New Points Count : ' + str(len(np.asarray(pcdcls.points))))


pcdclr, ind = pcdcls.remove_radius_outlier(nb_points=20, radius=0.05)
#display_inlier_outlier(pcdcls, ind)
print("\nRadius oulier removal\n")
print('Original Points Count : ' + str(len(np.asarray(pcdcls.points))))
print('New Points Count : ' + str(len(np.asarray(pcdclr.points))))

#o3d.visualization.draw_geometries([pcdclr])

print("\nPoint Cloud has Been Cleaned\n")
print("==========================================================================")


########## Both Method ##########

print("Start RANSAC Segmentation\n")

########## RANSAC Method ##########
segment_models={}
segments_RANSAC={}
max_plane_idx=4

rest=pcdclr
for i in range(max_plane_idx):
    #colors = plt.cm.get_cmap("tab20")(i)
    segment_models[i], inliers = rest.segment_plane(
        distance_threshold=0.08,
        ransac_n=3, num_iterations=15000)
    segments_RANSAC[i]=rest.select_by_index(inliers)
    #segments_RANSAC[i].paint_uniform_color(list(colors[:3]))
    rest = rest.select_by_index(inliers, invert=True)
    print("pass",i+1,"/",max_plane_idx,"done.")

#o3d.visualization.draw_geometries([segments_RANSAC[i] for i in range(max_plane_idx)])

#o3d.visualization.draw_geometries([rest])

print("RANSAC Segmentation is Finished \n")

#o3d.visualization.draw_geometries([segments_RANSAC[i] for i in range(max_plane_idx)])

#o3d.visualization.draw_geometries([rest])

#d_threshold = 0.03
#labels = np.array(segments[i].cluster_dbscan(eps=d_threshold*10, min_points=2))

print("==========================================================================")

#Data Cleaning
print("Start Cleaning Point Cloud\n")


pcdcls, ind = rest.remove_statistical_outlier(nb_neighbors=20,
                                                   std_ratio=1.1)
print("\nStatistical oulier removal\n")
#display_inlier_outlier(rest, ind)
#TT = pcd.select_by_index(ind, invert=True)
#TT2 = pcd.select_by_index(ind)
print('Original Points Count : ' + str(len(np.asarray(rest.points))))
print('New Points Count : ' + str(len(np.asarray(pcdcls.points))))

pcdclr, ind = pcdcls.remove_radius_outlier(nb_points=20, radius=0.03)
#display_inlier_outlier(pcdcls, ind)
print("\nRadius oulier removal\n")
print('Original Points Count : ' + str(len(np.asarray(pcdcls.points))))
print('New Points Count : ' + str(len(np.asarray(pcdclr.points))))

#o3d.visualization.draw_geometries([pcdclr])

rest = pcdclr
print("\nPoint Cloud has Been Cleaned\n")
print("==========================================================================")

print("Start DBSCAN Segmentation\n")

########## DBSCAN Method ##########
labels = np.array(rest.cluster_dbscan(eps=0.029, min_points=25))
max_label = labels.max()
print(f"point cloud has {max_label + 1} clusters")
#colors = plt.cm.get_cmap("tab10")(labels / (max_label if max_label > 0 else 1))
#colors[labels < 0] = 0
#rest.colors = o3d.utility.Vector3dVector(colors[:, :3])

# 340 Cluster
#o3d.visualization.draw_geometries([rest])


#o3d.visualization.draw_geometries([segments_RANSAC[i] for i in range(max_plane_idx)]+[rest])

print("DBSCAN Segmentation is Finished \n")

print("==========================================================================")

##Finding Different Segments

print("Start Sepearating Segments \n")

# Finding points and their value(which is their class or segment)
w, h = 2,len(labels)
My_segments = [[0 for x in range(w)] for y in range(h)] 

for i in range(len(labels)):
    My_segments[i] = (i,labels[i])

# Sorting and finding the count of each segment
S_My_segments = sorted(My_segments,key=lambda l:l[1], reverse=False)

#['hit','miss','miss','hit','miss'].count('hit')

valuee = []
counter = []
cc= -1
for i in range(len(S_My_segments)):
    c = S_My_segments[i][1]

    
    if c in valuee:
        counter[cc] += 1
    else:
        cc += 1
        valuee.append(c)
        counter.append(1)
        
Value_count = list(zip(valuee, counter))

Value_count = sorted(Value_count,key=lambda l:l[1], reverse=True)
 
# Finding point for each segment and poting them in a list
Number_of_pints = 200 #2048  # for finding those which has more than 'this number' points

segment_DBSCAN = []
segment_DBSCAN_others = []
for k in range(len(Value_count)):
    test_p = []
    if (Value_count[k][1] > Number_of_pints and Value_count[k][0] != 0 and Value_count[k][0] != -1 ) :
        for j in range(len(My_segments)):
            if My_segments[j][1] == Value_count[k][0]:
                test_p.append(My_segments[j][0])
        
        test_p = np.array(test_p)
        segment_DBSCAN.append(test_p)
    else:
        for j in range(len(My_segments)):
            if My_segments[j][1] == Value_count[k][0]:
                test_p.append(My_segments[j][0])
        
        test_p = np.array(test_p)
        segment_DBSCAN_others.append(test_p)
        
print("Number of segments with bigger than 100 points: " + str( len(segment_DBSCAN)))

print("\nSepearating Segments Have Been Finished\n")
print("==========================================================================")

#Cleaning Variables
del c, cc, counter, h, i, j, k, max_label, max_plane_idx, Number_of_pints, test_p, Value_count, valuee
del w, ind,inliers, labels, pcdclr, pcdcls ,S_My_segments, segment_models #,colors, segments

#*********IMPORTANT**********
#Selecting the segment for showing and doing what you want 

#10, 28, 33, 36, 38, 47
a = rest.select_by_index(segment_DBSCAN[16])
aa = a.farthest_point_down_sample(2048)
#o3d.visualization.draw_geometries([aa])
aaa = np.asarray(aa.points)
aaaa = aaa.reshape(1,2048, 3)
#o3d.visualization.draw_geometries([aa])








#%%
##########          Quality improvement          ##########
##########          Quality improvement          ##########
##########          Quality improvement          ##########
##########          Quality improvement          ##########
##########          Quality improvement          ##########

#%%
#Quality improvement putting point between other points in point cloud

def getpoints(X,_XX,_x,Y,_YY,_y,_z,_C):
    RX = []
    RY = []
    RZ = []
    RC = []
    for i in range(len(_XX)):
        if round(_XX[i]) == X:
            if round(_YY[i]) == Y:                    
                RX.append(_x[i])
                RY.append(_y[i])
                RZ.append(_z[i])
                RC.append(_C[i])
    return RX,RY,RZ,RC
def FindNewpoints0(_TX,_TY,_TZ,_TC):
    _X = np.sort(_TX) 
    _Y = np.sort(_TY) 
    _Z = np.sort(_TZ) 
    _NX,_NY,_NZ,_NC = [],[],[],[]
    for i in range(len(_X)-1):
        _NX.append((_X[i] + _X[i+1])/2)
        _NY.append((_Y[i] + _Y[i+1])/2)
        _NZ.append((_Z[i] + _Z[i+1])/2)
        _NC.append((_TC[i] + _TC[i+1])/2) 
    return np.asarray(_NX), np.asarray(_NY), np.asarray(_NZ), np.asarray(_NC)
def FindNewpoints(_TX,_TY,_TZ,_TC):  
    M = np.zeros((len(_TX),6))
    for i in range(len(_TX)):
        M[i][0] = _TX[i]
        M[i][1] = _TY[i]
        M[i][2] = _TZ[i]
        
        M[i][3] = _TC[i][0]
        M[i][4] = _TC[i][1]
        M[i][5] = _TC[i][2]
    MS = np.argsort(M[:, 2])
    MM = np.zeros((len(MS),6))
    for i in range(len(MS)):
        MM[i][0] = M[MS[i]][0]
        MM[i][1] = M[MS[i]][1]
        MM[i][2] = M[MS[i]][2]
        MM[i][3] = M[MS[i]][3]
        MM[i][4] = M[MS[i]][4]
        MM[i][5] = M[MS[i]][5]           
    _NX,_NY,_NZ,_NCR,_NCG, _NCB = [],[],[],[], [],[]
    for i in range(len(MM)-1):
        Checker1 = (MM[i][2]*100)-20
        Checker2 = (MM[i][2]*100)+20
        Tocheck = (MM[i+1][2]*100)
        if (Checker1 <= Tocheck) and (Tocheck <= Checker2) :    
            _NX.append((MM[i][0] + MM[i+1][0])/2)
            _NY.append((MM[i][1] + MM[i+1][1])/2)
            _NZ.append((MM[i][2] + MM[i+1][2])/2)
            
            _NCR.append((MM[i][3] + MM[i+1][3])/2)
            _NCG.append((MM[i][4] + MM[i+1][4])/2)
            _NCB.append((MM[i][5] + MM[i+1][5])/2)
    _NC = np.zeros((len(_NCB),3)) 
    for i in range(len(_NCR)):
        _NC[i][0] = _NCR[i]
        _NC[i][1] = _NCG[i]
        _NC[i][2] = _NCB[i] 
    return np.asarray(_NX), np.asarray(_NY), np.asarray(_NZ), np.asarray(_NC)
def Renew(_x,_y,X,x,Y,y,z,c):
    TX,TY,TZ,TC = getpoints(_x,X,x,_y,Y,y,z,c)
    X1,Y1,Z1,C1 = FindNewpoints(TX,TY,TZ,TC)
    return X1,Y1,Z1,C1
def Pointcreater(a,b,d,c):
    c = Colors
    x = a
    y = b
    z = d
    X = a*100
    X = np.round(X)
    XX = np.unique(X)
    Y = b*100
    Y = np.round(Y)
    YY = np.unique(Y)
    Z = d*100
    Z = np.round(Z)
    #ZZ = np.unique(Z)

    T = np.zeros((0,3))
    C = np.zeros((0,3))
    print(len(XX))
    for i in range(len(XX)):
        print(str(i))
        for j in range(len(YY)):
            #print(j)
            #X1,Y1,Z1,C1 = Renew(XX[i],YY[j])
            X1,Y1,Z1,C1 = Renew(XX[i],YY[j],X,x,Y,y,z,c)
            if len(X1 > 0):
                t = np.zeros((len(X1),3))
                #print(len(X1))
                for k in range(len(X1)):
                    t[k][0] = X1[k]
                    t[k][1] = Y1[k]
                    t[k][2] = Z1[k]
                    #print("aa")
                T = np.concatenate((T, t), axis=0)
                C = np.concatenate((C, C1), axis=0)
                
    return T,C



aa = rest.select_by_index(segment_DBSCAN[19])
o3d.visualization.draw_geometries([aa])

points = np.asarray(aa.points)
Colors = np.asarray(aa.colors)


a = points[:,0]
b = points[:,1]
d = points[:,2]
T2,C2 = Pointcreater(a,b,d,Colors) 
TT2 = np.zeros((len(T2),3))
TT2[:,0] = T2[:,0]
TT2[:,1] = T2[:,1]
TT2[:,2] = T2[:,2]

pcd.points = o3d.utility.Vector3dVector(TT2)
pcd.colors = o3d.utility.Vector3dVector(C2)
o3d.visualization.draw_geometries([pcd])
o3d.visualization.draw_geometries([pcd]+[aa])



a = points[:,1]
b = points[:,0]
d = points[:,2]
T2,C2 = Pointcreater(a,b,d,Colors)
TT2 = np.zeros((len(T2),3))
TT2[:,0] = T2[:,1]
TT2[:,1] = T2[:,0]
TT2[:,2] = T2[:,2]

pcd.points = o3d.utility.Vector3dVector(TT2)
pcd.colors = o3d.utility.Vector3dVector(C2)
#o3d.visualization.draw_geometries([pcd])
#o3d.visualization.draw_geometries([pcd]+[aa])

#%%








#%%
##########          3D object Detection Using PointNet          ##########
##########          3D object Detection Using PointNet          ##########
##########          3D object Detection Using PointNet          ##########
##########          3D object Detection Using PointNet          ##########
##########          3D object Detection Using PointNet          ##########
#%%



import os 
import glob 
import trimesh 
import numpy as np
import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import layers 
from matplotlib import pyplot as plt



DATA_DIR = os.path.join(os.path.dirname(r'D:\Keyhan-IDTC\ModelNet5\ModelNet5'), "ModelNet5")

mesh = trimesh.load(os.path.join (DATA_DIR, r"monitor\train\monitor_0010.off"))
#mesh.show ( )


points = mesh.sample(2048)
fig = plt.figure(figsize = (5, 5))
ax = fig.add_subplot (111, projection="3d")
ax.scatter(points [:,0], points[:,1], points[:,2])
ax.set_axis_off()
plt.show()


class_map = {}


folders = [ f.path for f in os.scandir(DATA_DIR) if f.is_dir() ]

#folders = glob.glob(os.path.join(DATA_DIR), recursive = True)




def parse_dataset(num_points=2048):
    
    train_points = []
    train_labels = []
    test_points = []
    test_labels = []
    class_map = {}
    #folders = glob.glob(os.path.join(DATA_DIR,"[!README]*"))
    folders = [ f.path for f in os.scandir(DATA_DIR) if f.is_dir() ]
    for i, folder in enumerate(folders):
        print("processing class: {}". format(os.path.basename(folder)))
        # store folder name with ID so we can retrieve later
        class_map[i] = folder.split("/")[-1]
        # gather all files
        train_files = glob.glob(os.path.join(folder,"train/*"))
        test_files = glob.glob(os.path.join(folder,"test/*"))
        
        for f in train_files:
            train_points.append(trimesh.load(f).sample(num_points))
            train_labels.append(i)
            
        
        for f in test_files:
            test_points.append(trimesh.load(f).sample(num_points))
            test_labels.append(i)
            
            
        #print(test_labels[len(test_labels)-2])
        #print(train_files[i])
        #print(train_labels[len(train_labels)-2])
        #print(class_map[i]) 
        
    return (
        np.array(train_points), 
        np.array(test_points),
        np.array(train_labels),
        np.array(test_labels),
        class_map,
    )

NUM_POINTS = 2048
NUM_CLASSES = 5
BATCH_SIZE = 32

train_points, test_points, train_labels, test_labels, CLASS_MAP = parse_dataset(NUM_POINTS)


def augment(points, label):
    # jitter points
    points += tf.random.uniform(points.shape, -0.005, 0.005, dtype = tf.float64)
    # shuffle points
    points = tf.random.shuffle(points)
    return points, label

train_dataset = tf.data.Dataset.from_tensor_slices((train_points, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_points, test_labels))

train_dataset = train_dataset.shuffle(len(train_points)).map(augment).batch(BATCH_SIZE)
#train_dataset = train_dataset.shuffle(len(train_points)).batch(BATCH_SIZE)
#train_dataset = augment(train_dataset[0])

test_dataset = test_dataset.shuffle(len(test_points)).batch(BATCH_SIZE)

def conv_bn(x, filters):
    x = layers.Conv1D(filters, kernel_size=1, padding="valid")(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)

def dense_bn(x, filters):
    x = layers.Dense(filters)(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)

class OrthogonalRegularizer(keras.regularizers.Regularizer):
    def __init__(self, num_features, l2reg=0.001):
        self.num_features = num_features
        self.l2reg = l2reg
        self.eye = tf.eye(num_features)
    def __call__(self, x):
        x = tf.reshape(x, (-1, self.num_features, self.num_features))
        xxt = tf.tensordot(x, x, axes=(2, 2))
        xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
        return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))

def tnet(inputs, num_features):
    
    # Initalise bias as the indentity matrix
    bias = keras.initializers.Constant(np.eye(num_features).flatten())
    reg = OrthogonalRegularizer(num_features)
    
    x = conv_bn(inputs, 32)
    x = conv_bn(x, 64)
    x = conv_bn(x, 512)
    x = layers.GlobalMaxPooling1D()(x)
    x = dense_bn(x, 256)
    x = dense_bn(x, 128)
    x = layers.Dense(
        num_features * num_features,
        kernel_initializer="zeros",
        bias_initializer=bias,
        activity_regularizer=reg,
    )(x)
    feat_T = layers.Reshape((num_features, num_features))(x)
    # Apply affine transformation to input features
    return layers.Dot(axes= (2, 1))([inputs, feat_T])


inputs = keras.Input(shape=(NUM_POINTS, 3))

x = tnet(inputs, 3)
x = conv_bn(x, 32)
x = conv_bn(x, 32)
x = tnet(x, 32)
x = conv_bn(x, 32) 
x = conv_bn(x, 64)
x = conv_bn(x, 512)
x = layers.GlobalMaxPooling1D()(x)
x = dense_bn(x, 256)
x = layers.Dropout(0.3)(x)
x = dense_bn(x, 128)
x = layers.Dropout(0.3)(x)

outputs = layers.Dense(NUM_CLASSES, activation= "softmax")(x)

model = keras.Model(inputs=inputs, outputs=outputs, name="pointnet")

model.summary()


model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer = keras.optimizers.Adam(learning_rate=0.01),
    metrics=["sparse_categorical_accuracy"],
)

model.fit(train_dataset, epochs=20, validation_data=test_dataset)


CLM = ""
for i in range(len(CLASS_MAP)) : CLM += str(i) + ":" +  CLASS_MAP[i].split("\\")[4] + "   "
print(CLM)
#%%
#*********IMPORTANT**********
#Selecting the segment for showing and doing what you want 

a = rest.select_by_index(segment_DBSCAN[17])
aa = a.farthest_point_down_sample(2048)
o3d.visualization.draw_geometries([aa])
aaa = np.asarray(aa.points)
aaaa = aaa.reshape(1,2048, 3)




#**********IMPORTANT**********
#Preparing point to predict : 


For_PredMy = tf.convert_to_tensor(aaaa)
# run test data through model
preds = model.predict(For_PredMy)
print(preds)
pred = tf.math.argmax(preds, -1)
print(np.asarray(pred))
checkpred = CLASS_MAP[pred[0].numpy()].split("\\")[4]
print(checkpred)



print("MAX : " , round(max(np.asarray(max(np.asarray(preds)))), 5))
print("MIN : " , round(min(np.asarray(max(np.asarray(preds)))), 5))



a = rest.select_by_index(segment_DBSCAN[40])
o3d.visualization.draw_geometries([a])

print(str(len(np.asarray(a.points))))
aa = a.farthest_point_down_sample(2048)
print(str(len(np.asarray(aa.points))))

#o3d.visualization.draw_geometries([aa])
aaa = np.asarray(aa.points)
aaaa = aaa.reshape(1,2048, 3)
o3d.visualization.draw_geometries([a])




For_PredMy = tf.convert_to_tensor(aaaa)
preds = model.predict(For_PredMy)
print(preds)
pred = tf.math.argmax(preds, -1)
print(np.asarray(pred))
checkpred = CLASS_MAP[pred[0].numpy()].split("\\")[4]
print(checkpred)



segment_DBSCAN_Prediction = []
for i in range(len(segment_DBSCAN)):
    print(str(i) , " / " , len(segment_DBSCAN))
    a = rest.select_by_index(segment_DBSCAN[i])
    aa = a.farthest_point_down_sample(2048)
    if(len(np.asarray(aa.points)) >= 2048):
        aaa = np.asarray(aa.points)
        aaaa = aaa.reshape(1,2048, 3)
        
        For_PredMy = tf.convert_to_tensor(aaaa)
        preds = model.predict(For_PredMy)
        pred = tf.math.argmax(preds, -1)
        segment_DBSCAN_Prediction.append(np.asarray(pred))
    else:
        segment_DBSCAN_Prediction.append(100)
    




segment_DBSCAN_Prediction = []
for i in range(20):
    print(str(i) , " / " , len(segment_DBSCAN))
    a = rest.select_by_index(segment_DBSCAN[i])
    aa = a.farthest_point_down_sample(2048)
    if(len(np.asarray(aa.points)) >= 2048):
        aaa = np.asarray(aa.points)
        aaaa = aaa.reshape(1,2048, 3)
        
        For_PredMy = tf.convert_to_tensor(aaaa)
        preds = model.predict(For_PredMy)
        pred = tf.math.argmax(preds, -1)
        segment_DBSCAN_Prediction.append(np.asarray(pred))
        checkpred = CLASS_MAP[pred[0].numpy()].split("\\")[4]
        print(checkpred)
    else:
        segment_DBSCAN_Prediction.append(100)
    print(np.asarray(pred))
    print(checkpred)
    print("MAX : " , round(max(np.asarray(max(np.asarray(preds)))), 5))
    print("MIN : " , round(min(np.asarray(max(np.asarray(preds)))), 5))
    #o3d.visualization.draw_geometries([aa])




AAB = []
POIN = []
for i in range( len(segment_DBSCAN)):
    a = rest.select_by_index(segment_DBSCAN[i])
    aabb = a.get_axis_aligned_bounding_box()
    aabb.color = (1, 0, 0)
    AAB.append(aabb)
    POIN.append(a)    
#%%
# showing The results for 3D object detection
o3d.visualization.draw_geometries([segments_RANSAC[i] for i in range(len(segments_RANSAC))] + # Walls
                                  [rest.select_by_index(segment_DBSCAN_others[i]) for i in range(len(segment_DBSCAN_others))]+ # Useless segments
                                  
                                  [POIN[i] for i in range(len(POIN))]+ # Used Segments for predicts
                                  [AAB[i] for i in range(len(AAB))] # bounderies
                                  )





#%%
##########          Converting 3D to 2D          ##########
##########          Converting 3D to 2D          ##########
##########          Converting 3D to 2D          ##########
##########          Converting 3D to 2D          ##########
##########          Converting 3D to 2D          ##########
#%%

## First approach by squizing points

def Convert_3D_TO_2D(PCD, Im_Size,Dim_Wanted, Direction):
    # Dim_Vanted is the dimention which is wanted for example: X and Y, or X and Z, or Y and Z
    # this parameter is string like 'xy' or 'xz' or 'yz'
    # Direction is the direction which we want to follow, for example from 0 to end or from last point to 0
    # it should be passed to the function like '01' or '10'
    points = np.asarray(PCD.points) # fetche the points
    colors = np.asarray(PCD.colors) # fetche the colors
    colors = (colors * 255).astype(np.uint8) # scaling the colors from 0 and 1 to 0 and 255
    
    # Project the points onto the XYZ plane
    xyz_points = points
    
    # Compute the range of the XYZ coordinates
    x_range = np.max(xyz_points[:, 0]) - np.min(xyz_points[:, 0])
    y_range = np.max(xyz_points[:, 1]) - np.min(xyz_points[:, 1])
    z_range = np.max(xyz_points[:, 2]) - np.min(xyz_points[:, 2])
    
    # Scale the XYZ coordinates to the range of 0 to 1
    xyz_points -= np.min(xyz_points, axis=0)
    xyz_points /= np.array([x_range, y_range, z_range])
    
    # Scale the XY coordinates to the size of the image
    image_size = (Im_Size, Im_Size,3)
    image_size_mul = (Im_Size, Im_Size, Im_Size)
    minn = np.min([x_range,y_range,z_range])
    if x_range == minn:
        image_size_mul = (Im_Size/2, Im_Size, Im_Size)
    if y_range == minn:
        image_size_mul = (Im_Size, Im_Size/2, Im_Size)
    if z_range == minn:
        image_size_mul = (Im_Size, Im_Size, Im_Size/2)

    xyz_points *= np.array(image_size_mul)
    
    # Rasterize the points into a 2D image
    image = np.zeros(image_size)
    xyz_points = xyz_points.astype(np.int32)
    
    # defining wanted dimention:
    a = 0
    b = 1
    c = 2
    zz = Im_Size/10
    if Dim_Wanted == 'xy':
        a = 0
        b = 1
        c = 2

    if Dim_Wanted == 'xz':
        a = 0
        b = 2
        c = 1

    if Dim_Wanted == 'yz':
        a = 1
        b = 2
        c = 0

    #o3d.visualization.draw_geometries([xyz_points])

    # finding the direction
    if Direction == '01':
        Z_Counter = 0
    else:
        Z_Counter  = Im_Size 
     
        
    # converting 3d to 2d
    counter = 0
    for i in range(Im_Size):
        if Direction == '01':
            Z_Counter += 1
        else:
            Z_Counter -= 1
        
        
        counter = 0
        for point in xyz_points:
            z = point[c].astype(np.int32)
            if z == Z_Counter :
                x = point[a].astype(np.int32)
                y = point[b].astype(np.int32)
                  
                if x <= Im_Size-1 and y <= Im_Size-1 and image[x,y,0] == 0 and image[x,y,1] == 0 and image[x,y,2] == 0 :
                    image[x, y,0] = colors[counter,0]
                    image[x, y,1] = colors[counter,1]
                    image[x, y,2] = colors[counter,2]
                    

                counter += 1   
                
               
    # for making background white
    for i in range(Im_Size):
        for j in range(Im_Size):
            if image[i,j,0] == 0 and image[i,j,1] == 0 and image[i,j,2] == 0 :
                image[i,j,0] = 255
                image[i,j,1] = 255
                image[i,j,2] = 255

    return image


a = rest.select_by_index(segment_DBSCAN[17])
im = Convert_3D_TO_2D(a,512,'yz','01')

image = im.astype(np.uint8)
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
plt.imshow(image)
plt.show()



#%%

## Second approach using scrinshot

a = rest.select_by_index(segment_DBSCAN[19])
#o3d.visualization.draw_geometries([a])

XX = [0,1,1.5,2]
YY = [0,1,1.5,2]
ZZ = [0,1,1.5,2]

#R = a.get_rotation_matrix_from_xyz(( np.pi / 1,  np.pi /10,  np.pi / 1)) #bala pain, Shift chaprast, chap va rast
R = a.get_rotation_matrix_from_xyz(( 0,  1,  -0.5)) #bala pain, Shift chaprast, chap va rast

#R will be 3x3 matrix by which the
#R will be in the form:
#R=[[ 0.99878604,  0.04768269,  0.01239592],
#   [ 0.04769243, -0.99886129, -0.00048715],
#   [ 0.01235845,  0.00107775, -0.99992352]]
a=a.rotate(R, center=(0,0,0))
#o3d.visualization.draw_geometries([a])


pcd = a
#vis = o3d.visualization.Visualizer()
vis = o3d.visualization.Visualizer()
vis.create_window()
vis.get_render_option().point_color_option = o3d.visualization.PointColorOption.Color
vis.get_render_option().point_size = 7.0
vis.add_geometry(pcd)
pathSave = "D:/Keyhan-IDTC/Test/file"+str(2)+".jpg"
vis.capture_screen_image(pathSave, do_render=True)
vis.destroy_window()



#python program to check if a directory exists
import os

Path = "D:/Keyhan-IDTC/Test/"+str(0)
isExist = os.path.exists(Path)
if not isExist:
    os.makedirs(Path)
      


XX = [1,-1.8,1.8,4]
YY = [1,-1.8,1.8,4]
ZZ = [1,-1.8,1.8,4]

#XX = [0,1,1.5,2]
#YY = [0,1,1.5,2]
#ZZ = [0,1,1.5,2]
for i in range(len(segment_DBSCAN)):
    counter = 0
    for x in range(len(XX)):
        for y in range(len(YY)):
            for z in range(len(YY)): 
                a = rest.select_by_index(segment_DBSCAN[19])
                #R = a.get_rotation_matrix_from_xyz(( np.pi /XX[x],  np.pi /YY[y],  np.pi / ZZ[z])) #bala pain, Shift chaprast, chap va rast
                R = a.get_rotation_matrix_from_xyz(( XX[x],  YY[y],  ZZ[z])) #bala pain, Shift chaprast, chap va rast
                a=a.rotate(R, center=(0,0,0))
                #o3d.visualization.draw_geometries([a])
                pcd = a
                vis = o3d.visualization.Visualizer()
                vis.create_window()
                vis.get_render_option().point_color_option = o3d.visualization.PointColorOption.Color
                vis.get_render_option().point_size = 7.0
                vis.add_geometry(pcd)
                Path = "D:/Keyhan-IDTC/Test/"+str(i)
                isExist = os.path.exists(Path)
                if not isExist:
                    os.makedirs(Path)
                PathSave = Path+"/"+str(counter)+".jpg"
                vis.capture_screen_image(PathSave, do_render=True)
                vis.destroy_window()
                counter += 1
            
    break



#%%
##########          2D Object Detection Using YOLO V3          ##########
##########          2D Object Detection Using YOLO V3          ##########
##########          2D Object Detection Using YOLO V3          ##########
##########          2D Object Detection Using YOLO V3          ##########
##########          2D Object Detection Using YOLO V3          ##########
#%%




import cv2
#import matplotlib as plt
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

classes = []

with open(r"C:\Users\jawad_pl\OneDrive - Teesside University\Project Lidar\Image detection\YOLO\COCO.names", 'r') as f:
    classes = f.read().splitlines()


Selected_classes = []

with open(r"C:\Users\jawad_pl\OneDrive - Teesside University\Project Lidar\Image detection\YOLO\COCO_SELECTED.names", 'r') as f:
    Selected_classes = f.read().splitlines()


yolo = cv2.dnn.readNet(r"C:\Users\jawad_pl\OneDrive - Teesside University\Project Lidar\Image detection\YOLO\yolov3.weights", r"C:\Users\jawad_pl\OneDrive - Teesside University\Project Lidar\Image detection\YOLO\Yolo3.cfg" )

Dirs = []
rootdir = r"D:\Keyhan-IDTC\Test"
for file in os.listdir(rootdir):
    d = os.path.join(rootdir, file)
    if os.path.isdir(d):
        Dirs.append(d)
        
Final_ClassName_ = []
Final_Image_ = []
Final_Prediction_ = []
Final_Confidence_ = []

for Dir in Dirs:
    CN = Dir.split("\\")
    ClassName = CN[len(CN)-1] 
    print(ClassName)
    ImagesNames = glob.glob(Dir + "/*")
    
    for imgpath in ImagesNames:
        
        IN = imgpath.split("\\")
        ImageName = IN[len(IN)-1] 
        
        img = cv2.imread(imgpath)
        blob = cv2.dnn.blobFromImage(img, 1/255, (320,320), (0,0,0), swapRB= True, crop = False)
        yolo.setInput(blob)
        output_layes_name = yolo.getUnconnectedOutLayersNames()
        layeroutput = yolo.forward(output_layes_name)
        
        width = img.shape[0]
        height = img.shape[1]
        
        boxes = []
        confidences = []
        class_ids = []
        
        for output in layeroutput :
            for detection in output:
                score = detection[10:]
                class_id = np.argmax(score)
                confidence = score[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_x = int(detection[0] * height)            
                    w = int(detection[0]*width)            
                    h = int(detection[0]*height) 
                    x = int(center_x - w/2)
                    y = int(center_x - h/2)
                    boxes.append([x,y,w,h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        
        if len(boxes) >= 2:
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            for i in indexes.flatten():
                label = str(classes[class_ids[i]])
                confi = str(round(confidences[i],2))

            if label in Selected_classes:
                #Final Results
                print(ClassName ," ", ImageName , " " , label , " " , confi)
                Final_ClassName_.append(ClassName)
                Final_Image_.append(ImageName)
                Final_Prediction_.append(label)
                Final_Confidence_.append(confi)
            





