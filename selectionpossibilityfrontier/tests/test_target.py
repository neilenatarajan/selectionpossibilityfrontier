import unittest
from selectionpossibilityfrontier.target import *

class TestFrontier(unittest.TestCase):

    def setUp(self) -> None:
        # Create some Dummy data to use in our testing
        total_size = 100
        select_size = 10
        proportion_female = 0.5
        proportion_white = 0.8
        proportion_low_ses = 0.4
        female = (np.random.rand(total_size) < proportion_female).astype(int)
        male = 1 - female

        white = (np.random.rand(total_size) < proportion_white).astype(int)
        bipoc = 1 - white

        low_ses = (np.random.rand(total_size) < proportion_low_ses).astype(int)

        score = np.random.rand(total_size)
        self.candidates = pd.DataFrame([male, female, white, bipoc, low_ses, score]).transpose()
        self.candidates.columns = ['Male', 'Female', 'White', 'BIPOC', 'Low_SES', 'Score']
        self.categories = self.candidates.drop('Score', axis=1).values.astype(float)
        self.scores = self.candidates['Score'].values.astype(float)

        self.weights = np.ones(self.categories.shape[1])
        self.proportions = np.array([.5, .5, 0.0, .4, .8])
        self.presences = np.array([0.0, 0.0, 0.0, 10, 10])
        self.select_size = 100

        self.proportional_target, self.proportional_tfmax = make_proportional_targets(self.select_size, self.proportions, self.weights, alpha=1)
        self.presence_target, self.presence_tfmax = make_presence_targets(self.presences, 1, alpha=1)
        self.target, self.tfmax = make_combined_targets((self.proportional_target, self.proportional_tfmax), (self.presence_target, self.presence_tfmax), scale=True)

        return super().setUp()

    def test_make_proportional_targets(self):
        self.assertEqual(self.proportional_tfmax, 50+50+0+40+80, 'The maximum value of the target function is wrong.')
        self.assertEqual(self.proportional_target(np.zeros((5))), 0, 'The target function yields the wrong value on the empty cohort.')

if __name__ == '__main__':
    unittest.main()