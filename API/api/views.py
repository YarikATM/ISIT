from rest_framework import generics
from rest_framework.response import Response
from api.ai.main import recognize
from api.serializers import RecognizeSerializer
from rest_framework import status
import numpy as np
import cv2
from api.models import DetectHistory


class DetectView(generics.GenericAPIView):
    serializer_class = RecognizeSerializer
    queryset = DetectHistory.objects.all()



    def post(self, request):
        try:
            serializer = self.get_serializer(data=request.data)
            if not serializer.is_valid():
                return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

            image = serializer.validated_data['image']
            print(serializer.validated_data)
            print(image)

            file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            result = recognize(img)
            return Response(result, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)




