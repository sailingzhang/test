# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

import faceproto_pb2 as faceproto__pb2


class FaceIdentyStub(object):
  # missing associated documentation comment in .proto file
  pass

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.HelloFace = channel.unary_unary(
        '/FaceIdenty.FaceIdenty/HelloFace',
        request_serializer=faceproto__pb2.HelloFaceReq.SerializeToString,
        response_deserializer=faceproto__pb2.HelloFaceRsp.FromString,
        )
    self.InitFace = channel.unary_unary(
        '/FaceIdenty.FaceIdenty/InitFace',
        request_serializer=faceproto__pb2.InitFaceReq.SerializeToString,
        response_deserializer=faceproto__pb2.InitFaceRsp.FromString,
        )
    self.Detect = channel.unary_unary(
        '/FaceIdenty.FaceIdenty/Detect',
        request_serializer=faceproto__pb2.DetectReq.SerializeToString,
        response_deserializer=faceproto__pb2.PredictRsp.FromString,
        )
    self.FindSimilarFace = channel.unary_unary(
        '/FaceIdenty.FaceIdenty/FindSimilarFace',
        request_serializer=faceproto__pb2.FindSimilarFaceReq.SerializeToString,
        response_deserializer=faceproto__pb2.FindSimilarFaceRsp.FromString,
        )
    self.FindSimilarHistoryFace = channel.unary_unary(
        '/FaceIdenty.FaceIdenty/FindSimilarHistoryFace',
        request_serializer=faceproto__pb2.FindSimilarFaceReq.SerializeToString,
        response_deserializer=faceproto__pb2.PredictRsp.FromString,
        )
    self.Compare = channel.unary_unary(
        '/FaceIdenty.FaceIdenty/Compare',
        request_serializer=faceproto__pb2.CompareReq.SerializeToString,
        response_deserializer=faceproto__pb2.CompareRsp.FromString,
        )
    self.Predict = channel.unary_unary(
        '/FaceIdenty.FaceIdenty/Predict',
        request_serializer=faceproto__pb2.PredictReq.SerializeToString,
        response_deserializer=faceproto__pb2.PredictRsp.FromString,
        )


class FaceIdentyServicer(object):
  # missing associated documentation comment in .proto file
  pass

  def HelloFace(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def InitFace(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def Detect(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def FindSimilarFace(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def FindSimilarHistoryFace(self, request, context):
    """rpc FindSimilarHistoryFace(FindSimilarFaceReq) returns (FindSimilarFaceRsp);
    """
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def Compare(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def Predict(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_FaceIdentyServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'HelloFace': grpc.unary_unary_rpc_method_handler(
          servicer.HelloFace,
          request_deserializer=faceproto__pb2.HelloFaceReq.FromString,
          response_serializer=faceproto__pb2.HelloFaceRsp.SerializeToString,
      ),
      'InitFace': grpc.unary_unary_rpc_method_handler(
          servicer.InitFace,
          request_deserializer=faceproto__pb2.InitFaceReq.FromString,
          response_serializer=faceproto__pb2.InitFaceRsp.SerializeToString,
      ),
      'Detect': grpc.unary_unary_rpc_method_handler(
          servicer.Detect,
          request_deserializer=faceproto__pb2.DetectReq.FromString,
          response_serializer=faceproto__pb2.PredictRsp.SerializeToString,
      ),
      'FindSimilarFace': grpc.unary_unary_rpc_method_handler(
          servicer.FindSimilarFace,
          request_deserializer=faceproto__pb2.FindSimilarFaceReq.FromString,
          response_serializer=faceproto__pb2.FindSimilarFaceRsp.SerializeToString,
      ),
      'FindSimilarHistoryFace': grpc.unary_unary_rpc_method_handler(
          servicer.FindSimilarHistoryFace,
          request_deserializer=faceproto__pb2.FindSimilarFaceReq.FromString,
          response_serializer=faceproto__pb2.PredictRsp.SerializeToString,
      ),
      'Compare': grpc.unary_unary_rpc_method_handler(
          servicer.Compare,
          request_deserializer=faceproto__pb2.CompareReq.FromString,
          response_serializer=faceproto__pb2.CompareRsp.SerializeToString,
      ),
      'Predict': grpc.unary_unary_rpc_method_handler(
          servicer.Predict,
          request_deserializer=faceproto__pb2.PredictReq.FromString,
          response_serializer=faceproto__pb2.PredictRsp.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'FaceIdenty.FaceIdenty', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))