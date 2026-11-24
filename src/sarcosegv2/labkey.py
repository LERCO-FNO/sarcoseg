from labkey.api_wrapper import APIWrapper
from loguru import logger as log


class LabkeyAccessor(APIWrapper):
    def __init__(
        self,
        domain,
        container_path,
        context_path=None,
        use_ssl=True,
        verify_ssl=True,
        api_key=None,
        disable_csrf=False,
        allow_redirects=False,
    ):
        super().__init__(
            domain,
            container_path,
            context_path,
            use_ssl,
            verify_ssl,
            api_key,
            disable_csrf,
            allow_redirects,
        )
