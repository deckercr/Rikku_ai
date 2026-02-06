from django.contrib import admin
from .models import UserProfile, FaceEmbedding


class FaceEmbeddingInline(admin.TabularInline):
    model = FaceEmbedding
    extra = 0
    readonly_fields = ('created_at',)
    fields = ('created_at',)


@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ('name', 'embedding_count', 'created_at')
    inlines = [FaceEmbeddingInline]

    def embedding_count(self, obj):
        return obj.face_embeddings.count()
    embedding_count.short_description = 'Faces'
