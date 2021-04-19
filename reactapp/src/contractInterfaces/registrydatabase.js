import web3 from './web3';
import {registry} from "../abi/abi";

const address = '0xD107156e3679FF0A9FC23a4eE8973Dd80b25A457';

export default new web3.eth.Contract(registry, address);