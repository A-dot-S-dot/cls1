from unittest import TestCase

from pde_solver.system_vector import SystemVector


class SimpleSystemVector(SystemVector):
    value = 0

    def _assemble(self):
        self.value += 1

    @SystemVector.assemble_before_call
    def __call__(self, *args):
        return self.value


class TestSystemVector(TestCase):
    def test_manual_assemble(self):
        vector = SimpleSystemVector()
        vector.assemble()

        self.assertEqual(vector.value, 1)
        self.assertEqual(vector(), 1)

    def test_automatic_assemble(self):
        vector = SimpleSystemVector()
        self.assertEqual(vector(), 1)

    def test_multiple_assemble_before_call(self):
        vector = SimpleSystemVector()
        vector.assemble()
        vector.assemble()

        self.assertEqual(vector.value, 1)

    def test_assemble_after_call(self):
        vector = SimpleSystemVector()
        vector()
        vector.assemble()

        self.assertEqual(vector.value, 2)

    def test_two_calls(self):
        vector = SimpleSystemVector()
        vector()
        self.assertEqual(vector(), 2)
