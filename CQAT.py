from main import *

### CLUSTERING ###
cluster_weights = tfmot.clustering.keras.cluster_weights
CentroidInitialization = tfmot.clustering.keras.CentroidInitialization

clustering_params = {
  'number_of_clusters': 16,
  'cluster_centroids_init': CentroidInitialization.KMEANS_PLUS_PLUS,
  'cluster_per_channel': True,
}

clustered_model = cluster_weights(model, **clustering_params)


clustered_model.compile(optimizer = "adam" , loss = 'binary_crossentropy' , metrics = ['accuracy'])

clustered_model.fit(train_gen,epochs=2,validation_data=test_gen)

stripped_clustered_model = tfmot.clustering.keras.strip_clustering(clustered_model)

# CQAT
quant_aware_annotate_model = tfmot.quantization.keras.quantize_annotate_model(
              stripped_clustered_model)
cqat_model = tfmot.quantization.keras.quantize_apply(
              quant_aware_annotate_model,
              tfmot.experimental.combine.Default8BitClusterPreserveQuantizeScheme())

cqat_model.compile(optimizer = "adam" , loss = 'binary_crossentropy' , metrics = ['accuracy'])

cqat_model.fit(train_gen,epochs=2,validation_data=test_gen)

converter = tf.lite.TFLiteConverter.from_keras_model(cqat_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
cqat_tflite_model = converter.convert()

save_tflite(cqat_tflite_model,'CQAT')