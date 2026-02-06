from django.db import models
from pgvector.django import VectorField


class UserProfile(models.Model):
    name = models.CharField(max_length=100, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name


class FaceEmbedding(models.Model):
    user = models.ForeignKey(
        UserProfile, on_delete=models.CASCADE, related_name='face_embeddings'
    )
    embedding = VectorField(dimensions=512)  # ArcFace produces 512-dim vectors
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.name} - face #{self.pk}"
