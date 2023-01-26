from vaetc.models import register_model

from .gwae import GromovWassersteinAutoEncoder
register_model("gwae", GromovWassersteinAutoEncoder)
