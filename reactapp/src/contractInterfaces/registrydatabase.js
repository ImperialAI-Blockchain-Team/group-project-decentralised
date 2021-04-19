import web3 from './web3';
import {registry} from "../abi/abi";

const address = '0xe8b4986eB204F62445bF8a04Bb533F6C7215AE70';

export default new web3.eth.Contract(registry, address);