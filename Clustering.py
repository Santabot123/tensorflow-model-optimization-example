from main import *


cluster_weights = tfmot.clustering.keras.cluster_weights
CentroidInitialization = tfmot.clustering.keras.CentroidInitialization

clustering_params = {
  'number_of_clusters': 16,
  'cluster_centroids_init': CentroidInitialization.LINEAR
}


clustered_model = cluster_weights(model, **clustering_params)

clustered_model.compile(optimizer = tf.keras.optimizers.Adam() , loss = 'binary_crossentropy' , metrics = ['accuracy'])

clustered_model.fit(train_gen,epochs=2,validation_data=test_gen)

export_clustered_model= tfmot.clustering.keras.strip_clustering(clustered_model)

converter = tf.lite.TFLiteConverter.from_keras_model(export_clustered_model)
tflite_clustered_model = converter.convert()
save_tflite(tflite_clustered_model,'clustered')