from main import *

### PRUNING ###
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

batch_size = 8
epochs = 3

end_step = len(train_gen) * epochs

pruning_params = {
      'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                               final_sparsity=0.95,
                                                               begin_step=0,
                                                               end_step=end_step)
}

model_for_pruning = prune_low_magnitude(model, **pruning_params)

model_for_pruning.compile(optimizer = "adam" , loss = 'binary_crossentropy' , metrics = ['accuracy'])

logdir = tempfile.mkdtemp()

callbacks = [
  tfmot.sparsity.keras.UpdatePruningStep(),
  tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
]

model_for_pruning.fit(train_gen,
epochs=2,
validation_data=test_gen,
callbacks=callbacks,
verbose=1)

pruned_model = tfmot.sparsity.keras.strip_pruning(model_for_pruning)

### QUANTIZATION(default) ###
converter = tf.lite.TFLiteConverter.from_keras_model(pruned_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_and_pruned_tflite_model = converter.convert()

save_tflite(quantized_and_pruned_tflite_model,'quantized_and_pruned')

