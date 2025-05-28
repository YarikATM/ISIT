from rest_framework import serializers








class RecognizeSerializer(serializers.Serializer):
    image = serializers.ImageField()