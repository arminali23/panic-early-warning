@app.middleware("http")
async def access_log_middleware(request: Request, call_next):
    rid = new_req_id()
    request.state.request_id = rid
    start = time.time()
    try:
        log_event(
            "request",
            request_id=rid,
            method=request.method,
            path=str(request.url.path),
            client=str(request.client.host if request.client else None),
        )
        resp = await call_next(request)
        dur = time.time() - start
        log_event(
            "response",
            request_id=rid,
            code=resp.status_code,
            duration_ms=int(dur * 1000),
            path=str(request.url.path),
        )
        # x-request-id header
        resp.headers["X-Request-ID"] = rid
        return resp
    except Exception as e:
        dur = time.time() - start
        log_event(
            "error",
            request_id=rid,
            error=str(e),
            duration_ms=int(dur * 1000),
            path=str(request.url.path),
        )
        raise
