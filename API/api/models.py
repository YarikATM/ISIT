from django.db import models








class LicensePlate(models.Model):
    signs = models.CharField(max_length=11)
    created_at = models.DateTimeField(auto_now_add=True)
    allow_to = models.BooleanField(default=False)

    def __str__(self):
        return self.signs


class DetectHistory(models.Model):
    license_plate = models.ForeignKey(LicensePlate, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)