import unittest
from colonyscopy import dummy

class TestDummy(unittest.TestCase):
	def test_dummy(self):
		self.assertEqual( dummy(), 42 )

# Boilerplate
if __name__ == "__main__":
	unittest.main(buffer=True)

