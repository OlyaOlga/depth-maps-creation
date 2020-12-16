import datetime
import time
import os

import tensorflow as tf
import numpy as np

import pix_2_pix.constants as cn
from pix_2_pix.loss import generator_loss, discriminator_loss
from pix_2_pix.architecture import Generator, Discriminator
from pix_2_pix.utils import generate_images, scale_up, convert_input_to_meters, convert_output_to_meters


class PipelinePix2Pix:

    checkpoint_dir = cn.CHECKPOINT_DIR
    checkpoint_prefix = cn.CHECKPOINT_PREFIX
    log_dir = cn.LOG_DIR

    def __init__(self):
        self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.generator = Generator()
        self.discriminator = Discriminator()
        self.generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

        self.summary_writer = tf.summary.create_file_writer(
            PipelinePix2Pix.log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                         discriminator_optimizer=self.discriminator_optimizer,
                                         generator=self.generator,
                                         discriminator=self.discriminator)

    def restore(self, path):
        ret = self.checkpoint.restore(path).expect_partial()
        print(ret)

    def test(self, test_ds):
        for example_input, example_target in test_ds.take(1):
            generate_images(self.generator, example_input, example_target)

    def evaluate(self, rgb, depth, batch_size=2):
        def compute_errors(gt, pred):
            thresh = np.maximum((gt / pred), (pred / gt))

            a1 = (thresh < 1.25).mean()
            a2 = (thresh < 1.25 ** 2).mean()
            a3 = (thresh < 1.25 ** 3).mean()

            abs_rel = np.mean(np.abs(gt - pred) / gt)

            rmse = (gt - pred) ** 2
            rmse = np.sqrt(rmse.mean())

            log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()

            return a1, a2, a3, abs_rel, rmse, log_10

        depth_scores = np.zeros((6, len(rgb)))  # six metrics

        bs = batch_size

        for i in range(len(rgb) // bs):
            x = rgb[(i) * bs:(i + 1) * bs, :, :, :]

            # Compute results
            true_y = depth[(i) * bs:(i + 1) * bs, :, :]

            pred_y = self.generator.predict(x, batch_size=bs)

            true_y = convert_output_to_meters(true_y)
            pred_y = convert_output_to_meters(pred_y)

            # Compute errors per image in batch
            for j in range(len(true_y)):
                errors = compute_errors(true_y[j], pred_y[j])

                for k in range(len(errors)):
                    depth_scores[k][(i * bs) + j] = errors[k]

        print(depth_scores)

        print(np.argmin(depth_scores, axis=1)[3:])
        print(np.argmax(depth_scores, axis=1)[:3])

        e = depth_scores.mean(axis=1)

        print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('a1', 'a2', 'a3', 'rel', 'rms', 'log_10'))
        print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(e[0], e[1], e[2], e[3], e[4], e[5]))



    @tf.function
    def train_step(self, input_image, target, epoch):
      with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = self.generator(input_image, training=True)

        disc_real_output = self.discriminator([input_image, target], training=True)
        disc_generated_output = self.discriminator([input_image, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target, self.loss_object)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output, self.loss_object)

      generator_gradients = gen_tape.gradient(gen_total_loss,
                                              self.generator.trainable_variables)
      discriminator_gradients = disc_tape.gradient(disc_loss,
                                                   self.discriminator.trainable_variables)

      self.generator_optimizer.apply_gradients(zip(generator_gradients,
                                              self.generator.trainable_variables))
      self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                  self.discriminator.trainable_variables))

      with self.summary_writer.as_default():
        tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
        tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
        tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)
        tf.summary.scalar('disc_loss', disc_loss, step=epoch)


    def fit(self, train_ds, epochs, test_ds):
      for epoch in range(epochs):
        start = time.time()

        if epoch%10==0:

            for example_input, example_target in test_ds.take(5):
                generate_images(self.generator, example_input, example_target, epoch)
        print("Epoch: ", epoch)

        # Train
        for n, (input_image, target) in train_ds.enumerate():
          print('.', end='')
          if (n+1) % 100 == 0:
            print()
          self.train_step(input_image, target, epoch)
        print()

        # saving (checkpoint) the model every 20 epochs
        if (epoch + 1) % 20 == 0:
          self.checkpoint.save(file_prefix = PipelinePix2Pix.checkpoint_prefix)

        print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                            time.time()-start))
      self.checkpoint.save(file_prefix = PipelinePix2Pix.checkpoint_prefix)
