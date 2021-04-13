import web3 from './web3';
import {registry} from "../abi/abi";

const address = '0xF7D53fa86f09508dcE9C70Bd813cE752fc50cDDF';

export default new web3.eth.Contract(registry, address);