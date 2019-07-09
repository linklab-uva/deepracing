# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: PacketCarTelemetryData.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


import PacketHeader_pb2 as PacketHeader__pb2
import CarTelemetryData_pb2 as CarTelemetryData__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='PacketCarTelemetryData.proto',
  package='deepf1.twenty_eighteen.protobuf',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n\x1cPacketCarTelemetryData.proto\x12\x1f\x64\x65\x65pf1.twenty_eighteen.protobuf\x1a\x12PacketHeader.proto\x1a\x16\x43\x61rTelemetryData.proto\"\xc0\x01\n\x16PacketCarTelemetryData\x12?\n\x08m_header\x18\x01 \x01(\x0b\x32-.deepf1.twenty_eighteen.protobuf.PacketHeader\x12M\n\x12m_carTelemetryData\x18\x02 \x03(\x0b\x32\x31.deepf1.twenty_eighteen.protobuf.CarTelemetryData\x12\x16\n\x0em_buttonStatus\x18\x03 \x01(\rb\x06proto3')
  ,
  dependencies=[PacketHeader__pb2.DESCRIPTOR,CarTelemetryData__pb2.DESCRIPTOR,])




_PACKETCARTELEMETRYDATA = _descriptor.Descriptor(
  name='PacketCarTelemetryData',
  full_name='deepf1.twenty_eighteen.protobuf.PacketCarTelemetryData',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='m_header', full_name='deepf1.twenty_eighteen.protobuf.PacketCarTelemetryData.m_header', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='m_carTelemetryData', full_name='deepf1.twenty_eighteen.protobuf.PacketCarTelemetryData.m_carTelemetryData', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='m_buttonStatus', full_name='deepf1.twenty_eighteen.protobuf.PacketCarTelemetryData.m_buttonStatus', index=2,
      number=3, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=110,
  serialized_end=302,
)

_PACKETCARTELEMETRYDATA.fields_by_name['m_header'].message_type = PacketHeader__pb2._PACKETHEADER
_PACKETCARTELEMETRYDATA.fields_by_name['m_carTelemetryData'].message_type = CarTelemetryData__pb2._CARTELEMETRYDATA
DESCRIPTOR.message_types_by_name['PacketCarTelemetryData'] = _PACKETCARTELEMETRYDATA
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

PacketCarTelemetryData = _reflection.GeneratedProtocolMessageType('PacketCarTelemetryData', (_message.Message,), {
  'DESCRIPTOR' : _PACKETCARTELEMETRYDATA,
  '__module__' : 'PacketCarTelemetryData_pb2'
  # @@protoc_insertion_point(class_scope:deepf1.twenty_eighteen.protobuf.PacketCarTelemetryData)
  })
_sym_db.RegisterMessage(PacketCarTelemetryData)


# @@protoc_insertion_point(module_scope)
