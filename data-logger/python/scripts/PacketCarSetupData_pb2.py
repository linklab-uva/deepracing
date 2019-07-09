# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: PacketCarSetupData.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


import PacketHeader_pb2 as PacketHeader__pb2
import CarSetupData_pb2 as CarSetupData__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='PacketCarSetupData.proto',
  package='deepf1.twenty_eighteen.protobuf',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n\x18PacketCarSetupData.proto\x12\x1f\x64\x65\x65pf1.twenty_eighteen.protobuf\x1a\x12PacketHeader.proto\x1a\x12\x43\x61rSetupData.proto\"\x99\x01\n\x12PacketCarSetupData\x12?\n\x08m_header\x18\x01 \x01(\x0b\x32-.deepf1.twenty_eighteen.protobuf.PacketHeader\x12\x42\n\x0bm_carSetups\x18\x02 \x03(\x0b\x32-.deepf1.twenty_eighteen.protobuf.CarSetupDatab\x06proto3')
  ,
  dependencies=[PacketHeader__pb2.DESCRIPTOR,CarSetupData__pb2.DESCRIPTOR,])




_PACKETCARSETUPDATA = _descriptor.Descriptor(
  name='PacketCarSetupData',
  full_name='deepf1.twenty_eighteen.protobuf.PacketCarSetupData',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='m_header', full_name='deepf1.twenty_eighteen.protobuf.PacketCarSetupData.m_header', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='m_carSetups', full_name='deepf1.twenty_eighteen.protobuf.PacketCarSetupData.m_carSetups', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
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
  serialized_start=102,
  serialized_end=255,
)

_PACKETCARSETUPDATA.fields_by_name['m_header'].message_type = PacketHeader__pb2._PACKETHEADER
_PACKETCARSETUPDATA.fields_by_name['m_carSetups'].message_type = CarSetupData__pb2._CARSETUPDATA
DESCRIPTOR.message_types_by_name['PacketCarSetupData'] = _PACKETCARSETUPDATA
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

PacketCarSetupData = _reflection.GeneratedProtocolMessageType('PacketCarSetupData', (_message.Message,), {
  'DESCRIPTOR' : _PACKETCARSETUPDATA,
  '__module__' : 'PacketCarSetupData_pb2'
  # @@protoc_insertion_point(class_scope:deepf1.twenty_eighteen.protobuf.PacketCarSetupData)
  })
_sym_db.RegisterMessage(PacketCarSetupData)


# @@protoc_insertion_point(module_scope)
