class SceneNode:
    def __init__(self):
        """
        Empty Scene Node constructor
        """
        pass
        
    def hit(self, origin, direction):
        """

        :param origin: Ray origin
        :param direction:  Ray direction
        :return: [has intersection, t, normal]
        """
        return None, [], []