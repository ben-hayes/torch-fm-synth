import math
import unittest

import torch

from torch_fm_synth.operator import Operator


class TestOperator(unittest.TestCase):
    def test_can_create(self):
        operator = Operator()
        self.assertIsInstance(operator, Operator)

    def test_is_nn_module(self):
        operator = Operator()
        self.assertIsInstance(operator, torch.nn.Module)

    def test_can_synthesise_sinusoid(self):
        dummy_sr = 2
        dummy_freq = 1
        sample_length = 4
        expected_output = torch.Tensor([1, -1, 1, -1])

        operator = Operator(sr=dummy_sr)
        actual_output = operator.forward(dummy_freq, length=sample_length)

        torch.testing.assert_allclose(actual_output, expected_output)

    def test_can_set_amplitude(self):
        dummy_sr = 4
        dummy_freq = 2
        dummy_amplitude = 0.5
        sample_length = 4
        expected_output = torch.Tensor([0.5, -0.5, 0.5, -0.5])

        operator = Operator(sr=dummy_sr)
        actual_output = operator.forward(
            dummy_freq,
            amplitude=dummy_amplitude,
            length=sample_length)

        torch.testing.assert_allclose(actual_output, expected_output)

    def test_can_set_tensor_amplitude(self):
        dummy_sr = 8
        dummy_freq = 4
        dummy_amplitude = torch.Tensor([0, 0.4, 0.9, 13.0])
        sample_length = 4
        expected_output = torch.Tensor([0.0, -0.4, 0.9, -13.0])

        operator = Operator(sr=dummy_sr)
        actual_output = operator.forward(
            dummy_freq,
            amplitude=dummy_amplitude,
            length=sample_length)

        torch.testing.assert_allclose(actual_output, expected_output)

    def test_can_set_tensor_frequency(self):
        dummy_sr = 8
        dummy_freq = torch.Tensor([2, 0, 4, 2])
        sample_length = 4
        expected_output = torch.Tensor([1, 1, -1, 0])

        operator = Operator(sr=dummy_sr)
        actual_output = operator.forward(
            dummy_freq,
            length=sample_length)

        torch.testing.assert_allclose(actual_output, expected_output)

    def test_throws_if_scalar_freq_and_amp_with_no_length(self):
        dummy_sr = 8
        dummy_freq = 3
        dummy_amp = 0.7

        operator = Operator(sr=dummy_sr)
        with self.assertRaises(ValueError):
            operator.forward(dummy_freq, dummy_amp)

    def test_throws_if_tensor_freq_and_amp_are_different_lengths(self):
        dummy_sr = 8
        dummy_freq = torch.Tensor([0.5, 0.1, 0.3])
        dummy_amp = torch.Tensor([12, 0.1, 0.2, 0.5])

        operator = Operator(sr=dummy_sr)
        with self.assertRaises(ValueError):
            operator.forward(dummy_freq, dummy_amp)

    def test_throws_if_tensor_freq_and_amp_have_more_than_batch_and_time_dims(
            self):
        dummy_sr = 8
        dummy_freq = torch.Tensor([[[0.1]]])
        dummy_amp = torch.Tensor([[[0.5]]])

        operator = Operator(sr=dummy_sr)
        with self.assertRaises(ValueError):
            operator.forward(dummy_freq, dummy_amp)

    def test_can_set_initial_phase(self):
        dummy_sr = 4
        dummy_freq = 1
        dummy_amp = 0.8
        dummy_initial_phase = 0.5 * math.pi
        sample_length = 4
        expected_output = torch.Tensor([0, -0.8, 0, 0.8])

        operator = Operator(sr=dummy_sr)
        actual_output = operator.forward(
            dummy_freq,
            dummy_amp,
            phase_mod=dummy_initial_phase,
            length=sample_length)

        torch.testing.assert_allclose(actual_output, expected_output)

    def test_can_modulate_phase(self):
        dummy_sr = 4
        dummy_freq = 1
        dummy_phase_mod = torch.Tensor([math.pi, 0, -math.pi, math.pi])
        sample_length = 4
        expected_output = torch.Tensor([-1, 0, 1, 0])

        operator = Operator(sr=dummy_sr)
        actual_output = operator.forward(
            dummy_freq,
            phase_mod=dummy_phase_mod,
            length=sample_length)

        torch.testing.assert_allclose(actual_output, expected_output)

    def test_throws_if_phase_mod_has_more_than_batch_and_time_dims(self):
        dummy_sr = 8
        dummy_freq = 4
        dummy_amp = 0.5
        dummy_phase = torch.Tensor([[[0.1]]])

        operator = Operator(sr=dummy_sr)
        with self.assertRaises(ValueError):
            operator.forward(
                dummy_freq,
                dummy_amp,
                length=4,
                phase_mod=dummy_phase)

    def test_can_use_amp_with_multiple_batches_but_one_time_step(self):
        dummy_sr = 4
        dummy_freq = 1
        dummy_amp = torch.Tensor([[1], [0.5]])
        sample_length = 3
        expected_output = torch.Tensor([[1, 0, -1], [0.5, 0, -0.5]])

        operator = Operator(sr=dummy_sr)
        actual_output = operator.forward(
            dummy_freq,
            dummy_amp,
            length=sample_length)

        torch.testing.assert_allclose(actual_output, expected_output)

    def test_can_use_freq_with_multiple_batches_but_one_time_step(self):
        dummy_sr = 4
        dummy_freq = torch.Tensor([[1], [0]])
        dummy_amp = 1
        sample_length = 3
        expected_output = torch.Tensor([[1, 0, -1], [1, 1, 1]])

        operator = Operator(sr=dummy_sr)
        actual_output = operator.forward(
            dummy_freq,
            dummy_amp,
            length=sample_length)

        torch.testing.assert_allclose(actual_output, expected_output)

    def test_can_use_phase_with_multiple_batches_but_one_time_step(self):
        dummy_sr = 4
        dummy_freq = 1
        dummy_amp = 1
        dummy_phase = torch.Tensor([[0], [math.pi]])
        sample_length = 3
        expected_output = torch.Tensor([[1, 0, -1], [-1, 0, 1]])

        operator = Operator(sr=dummy_sr)
        actual_output = operator.forward(
            dummy_freq,
            dummy_amp,
            length=sample_length,
            phase_mod=dummy_phase)

        torch.testing.assert_allclose(actual_output, expected_output)
