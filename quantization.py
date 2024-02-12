from main import *

### DEFAULT OPTIMIZATION ###

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter.convert()

save_tflite(tflite_quant_model,'quant default')


### DEFAULT FULL INT QUANTIZATION ###
def representative_data_gen():
    for i in range(50):
        image=np.array(test_gen[i][0][0],dtype=np.float32)
        # tf.constant
        image=np.expand_dims(image,axis=0)
        image=tf.constant(image,dtype=tf.float32)
        yield [image]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen

converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

tflite_model_quant = converter.convert()

save_tflite(tflite_model_quant,'int_only')


### QUANTIZATION AWARE TRAINING ###

quant_aware_model = tfmot.quantization.keras.quantize_model(model)

quant_aware_model.compile(optimizer = "adam" , loss = 'binary_crossentropy' , metrics = ['accuracy'])
quant_aware_model.summary()

quant_aware_model.fit(train_gen, epochs=3,validation_data=test_gen)

converter = tf.lite.TFLiteConverter.from_keras_model(quant_aware_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

quantized_tflite_model = converter.convert()

save_tflite(quantized_tflite_model,'quant_aware')