from main import *

### PRUNING ####
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

batch_size = 8
epochs = 2

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

stripped_pruned_model = tfmot.sparsity.keras.strip_pruning(model_for_pruning)

### PQAT ###

quant_aware_annotate_model = tfmot.quantization.keras.quantize_annotate_model(
              stripped_pruned_model)
pqat_model = tfmot.quantization.keras.quantize_apply(
              quant_aware_annotate_model,
              tfmot.experimental.combine.Default8BitPrunePreserveQuantizeScheme())

pqat_model.compile(optimizer = "adam" , loss = 'binary_crossentropy' , metrics = ['accuracy'])

pqat_model.fit(train_gen,epochs=2,validation_data=test_gen)

converter = tf.lite.TFLiteConverter.from_keras_model(pqat_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
pqat_tflite_model = converter.convert()

save_tflite(pqat_tflite_model,'PQAT')

